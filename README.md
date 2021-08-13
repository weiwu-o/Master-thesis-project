# Master-thesis-project
Sequential Anomaly Detection for Log Data Using Deep Learning

**This is a cooperative project with Volvo GTT and the goal is to do effective anomaly detection for company log data, however, we will not show any company data since it needs to be confidential.  We also used a public dataset to evaluate our methods which is the part we are going to present here.**


The main sourse is a articel from logPAI: <https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf>

And we uesed their data (HDFS) and build a model:

- Unidirecitonal Long short-term Memory (Uni-LSTN)

Next we have an extend version model:

- Bidirectional Long short-term Memory (Bi-LSTM)

However, the above models need a threshold which will plus an necessary extra step. Then we tried: 

- Seq2seq 

and

- Transformer

**Conclusion**

Uniditrctional Long short-term Memory works best for both Volov data and HDFS data.
