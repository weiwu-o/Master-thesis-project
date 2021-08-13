```mermaid
graph LR
   A[Data] --> B[Data prepricessing]

   
   
   B -- Use Pytorch -->E[Create windows]
   E -->EE[Train Seq2seq model]
   EE --> EEE[Anomaly Detection]

``` 

