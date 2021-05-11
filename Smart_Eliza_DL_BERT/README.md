# Smart-Eliza chatbot in Python using goemotion and Bert model


- To generate New BERT model using GoEmotions dataset,

Use below command with parameters shows as below:
 “python bert_classifier.py --vocab_file=C:\CIT_master_paper_important\BERT-Base-model_uncased_L-12_H-768_A-12\vocab.txt --bert_config_file=C:\CIT_master_paper_important\BERT-Base-model_uncased_L-12_H-768_A-12\bert_config.json --output_dir=C:\CIT_master_paper_important\output_dir”

(Becuase the New BERT model that genertaed from the above command is about 400MB which is over github size limit, so please generat this model file first locally , it might cost 20 hours run to complete.)

- To run the Smart-ELIZA using the command :python eliza_with_ml_bert.py


- The other reference link is as blow:

 Original Eliza code from the github https://github.com/nlpia/eliza

 Bert code from github https://github.com/google-research/bert
 
 Goemotion DB: https://github.com/google-research/google-research/tree/master/goemotions
 
 
