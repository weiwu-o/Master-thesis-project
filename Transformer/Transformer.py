import torch
import torch.nn as nn
import pandas as pd
import torch.utils.data as data
import sys
import re

from time import time

######################################################################
############ Create the dataframe for PyTorch input
######################################################################

def create_windows(log_key_sequence, window_size, step):
    windows=[]
    for i in range(0, len(log_key_sequence) - window_size+1, step):
        sentence = log_key_sequence[i:i + window_size]
        windows.append(sentence)
    return windows

def create_input_Transformer(df, window_size, step):
    split_data=[]
    for item in range(len(df)):
        split_data.extend(create_windows(df.iloc[item].EventSequence, window_size, step))
    data_split=pd.Series(split_data)
    df_transformer=pd.DataFrame(data_split, columns=['Windows'])
    return df_transformer['Windows']


######################################################################
############ Splitting at 'command'-line
######################################################################

#### Split the entire eventsequence (log integer sequence) on the "sendcommand"-string
def split_at_command(preprocessed_file, word2id, value=2):
    #sequence=[word2id.get(preprocessed_file[i],0) for i in range(len(preprocessed_file))]
    file=[]
    indices=[i for i, x in enumerate(preprocessed_file) if x==value]
    for start, end in zip([0, *indices], [*indices, len(preprocessed_file)]):
        file.append(preprocessed_file[start:end])
    return file[1:]

#### Split the original log on the "SendCommand"-lines
def find_commands(log):
    #file=file.decode('utf-8')
    log=log.splitlines()
    
    send_commands=[re.search(' SendCommand', log[i]) for i in range(len(log))]
    
    indices=[i for i, j in enumerate(send_commands) if j is not None]
    
    blocks=[]
    for k in range(1,len(indices)):
        blocks.append(log[indices[k-1]:indices[k]])
    return blocks

######################################################################
############ Classes for model
######################################################################

class SelfAttention(nn.Module):
    # embedding size and heads (heads=parts we split the embedding in)
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size=embed_size
        self.heads=heads
        
        # integer division below
        self.head_dim=embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"
        
        self.values=nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys=nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries=nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)
        
    def forward(self, values, keys, query, mask):
        # training samples has dim (N, sentence_length)
        N = query.shape[0] # no of training samples sent in at the same time
        # below corresponnds to source sentence length and target sentence length
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1] 
        
        ### Split embedding into self.heads pieces
        values=values.reshape(N, value_len, self.heads, self.head_dim)
        keys=keys.reshape(N, key_len, self.heads, self.head_dim)
        queries=query.reshape(N, query_len, self.heads, self.head_dim)
        
        values=self.values(values)
        keys=self.keys(keys)
        queries=self.queries(queries)
        
        ### Multiply queries with the keys
        energy=torch.einsum("nqhd, nkhd->nhqk", [queries, keys]) # n=batch_size, q=query_len, h=heads_len, d= 
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len = source sentence, heads, heads_dim)
        # energy shape: (N, heads, query_len=target sentence, key_len)
        
        if mask is not None:
            energy=energy.masked_fill(mask==0, float("-1e20")) #1e-20 basically infinity
            
        ### now run this through a softmax (Attention(Q, K, V)=softmax((QT^T)/sqrt(d_k))V from "Attention is all you need")
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        attention=torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        # we want (N, query_len, heads, heads_dim)
        out=torch.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(
        N, query_len, self.heads*self.head_dim
        )
        
        out=self.fc_out(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        
        ### SelfAttention we defined above
        self.attention=SelfAttention(embed_size, heads)
        
        ### LayerNorm
        self.norm1=nn.LayerNorm(embed_size)
        self.norm2=nn.LayerNorm(embed_size)
        
        self.feed_forward=nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size), # in paper forward_expansion=4
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        
        self.dropout=nn.Dropout(dropout)
        
    def forward(self, value, key, query, mask):
        attention=self.attention(value, key, query, mask)
        x=self.dropout(self.norm1(attention + query)) # send in a skip connection
        forward=self.feed_forward(x)
        out=self.dropout(self.norm2(forward + x))
        return out
    
class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length):
        super(Encoder, self).__init__()
        self.embed_size=embed_size
        self.device=device
        self.word_embedding=nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding=nn.Embedding(max_length, embed_size)
        
        self.layers= nn.ModuleList(
            [
                TransformerBlock(
                    embed_size, heads, dropout=dropout, forward_expansion=forward_expansion
                ) 
                for _ in range(num_layers)
            ]
        )
        self.dropout=nn.Dropout(dropout)
        
        
    def forward(self, x, mask):
        N, seq_length=x.shape
        positions=torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        out=self.dropout(self.word_embedding(x)+self.position_embedding(positions))
        
        for layer in self.layers:
            # special case in encoder, all inputs the same
            out=layer(out, out, out, mask)
            
            return out
        
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention=SelfAttention(embed_size, heads)
        self.norm=nn.LayerNorm(embed_size)
        self.transformer_block =TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout=nn.Dropout(dropout)
        
    def forward(self, x, value, key, src_mask, trg_mask):
        attention=self.attention(x, x, x, trg_mask)
        query =self.dropout(self.norm(attention+x))
        out=self.transformer_block(value, key, query, src_mask)
        return out
    
class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device=device
        self.word_embedding=nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding=nn.Embedding(max_length, embed_size)
        
        self.layers=nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
            for _ in range(num_layers)]
        )
        
        self.fc_out=nn.Linear(embed_size, trg_vocab_size)
        self.dropout=nn.Dropout(dropout)
        
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length=x.shape
        positions=torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x=self.dropout((self.word_embedding(x)+self.position_embedding(positions)))
        
        for layer in self.layers:
            x=layer(x, enc_out, enc_out, src_mask, trg_mask)
            
        out=self.fc_out(x) # prediction on which word is next
        return out
        
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx,device,  embed_size=256, num_layers=6, 
                forward_expansion=4, heads=8, dropout=0, max_length=11):
        super(Transformer, self).__init__()
        
        self.encoder=Encoder(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        self.decoder=Decoder(trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length)
        self.src_pad_idx=src_pad_idx
        self.trg_pad_idx=trg_pad_idx
        self.device=device
        
    def make_src_mask(self, src):
        src_mask =(src!= self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        N, trg_len=trg.shape
        trg_mask= torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len) #triangular matrix
        
        return trg_mask.to(self.device)
    
    ### forward:
    # src: sequence of log keys of size window size
    # trg: same as src but shifted right and sos-token in the beginning
    # EXAMPLE: if src=[1, 2, 3] then trg=['sos_token', 1, 2]
    def forward(self, src, trg):
        src_mask=self.make_src_mask(src)
        trg_mask=self.make_trg_mask(trg)
        enc_src=self.encoder(src, src_mask)
        out=self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

    ### evaluate: 
    # src: sequence of log keys of size window size
    # sos_token: the chosen log key used for start of sentence initialization
    def evaluate(self, src, sos_token):
        src_mask=self.make_src_mask(src)
        enc_src=self.encoder(src, src_mask)
        #output=[sos_token]
        start_of_string_t = torch.tensor(src.shape[0]*[sos_token]).view(-1,1)
        #output2=torch.LongTensor(output).unsqueeze(0)
        
        for i in range(10):
            
          #trg_mask=self.make_trg_mask(output2)
          trg_mask=self.make_trg_mask(start_of_string_t)
            
          out=self.decoder(start_of_string_t, enc_src, src_mask, trg_mask)

          start_of_string_t=torch.cat((start_of_string_t, torch.argmax(out, dim=2)[:,-1].view(-1,1)), dim=1)

          #output2=torch.tensor(output).unsqueeze(0)
        
        return start_of_string_t


######################################################################
############ Creating the dataloader
######################################################################

class Dataset(data.Dataset):
    def __init__(self, encoder_series):
        self.encoder_series=encoder_series
        self.num_total_seqs=len(self.encoder_series)
        
    def __len__(self):
        return self.num_total_seqs
        
    def __getitem__(self, index):
        encoder_input=self.encoder_series.iloc[index]
        
        return torch.LongTensor(encoder_input) #.cuda()
    
def dataloader_(encoder_series, batch_size):
    dataset=Dataset(encoder_series)
    
    data_loader=torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                           shuffle=False, drop_last=True)
    return data_loader


######################################################################
############ Return model
######################################################################

def Transformer_model(word2id):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    src_pad_idx=word2id['<pad>']
    trg_pad_idx=word2id['<pad>']
    src_vocab_size=len(word2id)
    trg_vocab_size=len(word2id)
    model=Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device)
    return model

######################################################################
############ Train model
######################################################################

def train_Transformer(model, train_loader, test_loader, epochs, word2id, window_size):
    start_time=time()

    sos_token=word2id['<sos>']

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr = 1e-3, betas=(0.9, 0.98), eps=1e-09, amsgrad=True)

    # Specify loss function
    loss_fn = nn.CrossEntropyLoss()

    losses=[]
    models=[]

    # iterate over amount of epochs
    for epoch in range(epochs):
    
    # accuracy is 0 in the beginning of training
        acc=0
    
    # iterate over batches in the dataloader for the training data set
        for b in train_loader:
      # set gradients to zero before backpropagation
            model.zero_grad()
        
      # send the data from the dataloader to the device (gpu or cpu)  
            x = b.to(device)
            y = x.to(device)

      # create input to the decoder (starts with the sos_token)
            start_=torch.tensor(x.shape[0]*[sos_token]).view(1, -1).to(device)
            x_dec=torch.cat((start_, x[:,:-1].t())).t().to(device)

      # forward pass  
            out = model.forward(x, x_dec).to(device)

      #reshape output and targets to be used in the accuracy and loss calculation
            out = out.reshape(out.shape[0]*out.shape[1],-1)
            targets = y.reshape(y.shape[0]*y.shape[1], -1)
      
      # compute accuracy (correct predictions/all predictions)  
            accuracy=torch.sum(torch.argmax(out, dim=1)==targets.squeeze(1))/out.size(0) 
      
      # accumulate accuracy
            acc+=accuracy.item()
    
      # compute loss
            loss = loss_fn(out, targets.squeeze(1))
        
      # compute gradient of loss function wrt all parameters with requires_grad=True  
            loss.backward()
        
      # update value of all parameters using the gradient from "loss.backward()"  
            opt.step()
        
    # Evaluation of the model (here no updates to the parameters are made)
        with torch.no_grad():
            for inp in test_loader:
        
                acc2=0
        
            # Validation accuracy
                out=model.evaluate(inp, word2id['<sos>'])
                accuracy=torch.sum(out[:,1:]==inp)/window_size 
                acc2+=accuracy.item()
                print('Validation accuracy: {}'.format(acc2/len(inp)))
    
        # Print training loss and accuracy per epoch
        print('Epoch {} Loss {}, Accuracy {}'.format(epoch, loss.item(), acc/len(train_loader)))
    
        # in the first epoch we save the loss and the model state dict
        if epoch==0:
            losses.append(loss.item())
            models.append(model.state_dict())
        
        # if loss of the previously saved smallest loss is larger than the current loss    
        elif losses[-1]>loss.item():
            losses.append(loss.item())
            models.append(model.state_dict())
    
    tot_time=time()-start_time
    print(tot_time/(60*60))

    path='Transformer\\trained_models\Transformer_ETL'
    torch.save(models[-1], path)

######################################################################
############ Anomaly detection
######################################################################

def anomaly_detection_evaluation(anomaly_df, path, word2id, window_size, step):
    TP=0
    FP=0
    TN=0
    FN=0

    anomaly_df['Blocks']=anomaly_df['EventSequence'].map(lambda x: split_at_command(x, word2id))

    model=Transformer_model(word2id=word2id)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    for s in range(len(anomaly_df)):
        sequence=anomaly_df.Blocks.iloc[s]
        label=anomaly_df.Label.iloc[s]
        file_input=[]
        for block in sequence:
            file_input.extend(create_windows(block, window_size, step))
        
        file_input=pd.Series(file_input)
        file_df=pd.DataFrame(file_input, columns=['Windows'])
        file=file_df['Windows']
    
        dataloader_file=dataloader_(file, len(file))
    
        for k in dataloader_file:
            preds=model.evaluate(k, word2id['<sos>'])
            accuracy=torch.sum(preds[:,1:]==k)/(len(file)*10)
            print('Accuracy {}, Label {}'.format(accuracy, label))
        
        if int(accuracy)!=1 and label==1:
            TP+=1
        elif int(accuracy)!=1 and label==0:
            FP+=1
        elif int(accuracy) == 1 and label ==1:
            FN+=1
        else:
            TN+=1
    sys.stdout = open("Transformer/results/results_Transformer.txt", "w")    
    print("True positive:")
    print(TP)
    print("False positive:")
    print(FP)
    print("False negative:")
    print(FN)
    print("True negative:")
    print(TN)

    accuracy=(TP+TN)/(TP+TN+FP+FN)
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    f1_score=(2*recall*precision)/(recall+precision)

 

    print("Accuracy:",accuracy)
    print("Precision:",precision)
    print("Recall:",recall)
    print("F1 score:",f1_score)
    sys.stdout.close()

def anomaly_detection(anomaly_df, path, word2id, id2word, window_size, step):
    sys.stdout = open("Transformer/results/anomaly_detection_Transformer.txt", "w") 
    anomaly_df['Blocks']=anomaly_df['EventSequence'].map(lambda x: split_at_command(x, word2id))

    model=Transformer_model(word2id=word2id)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    #model=Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device)
    #model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    for s in range(len(anomaly_df)):
        print('###################################################################################')
        print('File address: '+anomaly_df['address'].iloc[s]+ '\\' +anomaly_df['Filename'].iloc[s])
        original_log_blocks=find_commands(anomaly_df.Log.iloc[s])
        sequence=anomaly_df.Blocks.iloc[s]
        file_input=[]
        used_blocks=[]
        for block in range(len(original_log_blocks)):
            if len(original_log_blocks[block])<11:              
                print(original_log_blocks[block])
                print('------------------------------------------')
            else:
                used_blocks.append(block)

        sequence=[anomaly_df.Blocks.iloc[s][b] for b in used_blocks]
        for block in range(len(sequence)):
            file_input=create_windows(sequence[block], window_size, step)
        
            file_input=pd.Series(file_input)
            file_df=pd.DataFrame(file_input, columns=['Windows'])
            file=file_df['Windows']
    
            dataloader_file=dataloader_(file, len(file))
    
            for k in dataloader_file:
        #print(k.size())
                preds=model.evaluate(k, word2id['<sos>'])
        #print(preds[:,1:].size())
            accuracy=torch.sum(preds[:,1:]==k)/(len(file)*10)
                
        
            if int(accuracy)!=1:
                [print(original_log_blocks[block][i].encode('utf-8')) for i in range(len(original_log_blocks[block]))]
            else:
                continue

            print('---------------------------------------------------------------------------')
       
    sys.stdout.close()