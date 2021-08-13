```mermaid
graph LR
   A[Data] --> B[Data prepricessing]

   B -- Use Pytorch-->F[Create windows]
   F -->FF[Train Transformer model]
   FF --> FFF[Anomaly Detection]
``` 

