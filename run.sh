python3 train.py    #yelp att_cnn acc
python3 train.py -co_variant 0 # variant: no neighbors
python3 train.py -ui_variant 0 # variant: no user_item interaction 
python3 train.py -base_model cnn # variant: no attention over the tokens

python3 train.py -orient rmse #yelp att_cnn acc
python3 train.py -co_variant 0 -orient rmse # variant: no neighbors
python3 train.py -ui_variant 0 -orient rmse # variant: no user_item interaction 
python3 train.py -base_model cnn -orient rmse # variant: no attention over the tokens




python3 train.py -domain imdb     #yelp att_cnn acc
python3 train.py -co_variant 0 -domain imdb  # variant: no neighbors
python3 train.py -ui_variant 0 -domain imdb  # variant: no user_item interaction 
python3 train.py -base_model cnn -domain imdb  # variant: no attention over the tokens

python3 train.py -orient rmse -domain imdb  #yelp att_cnn acc
python3 train.py -co_variant 0 -orient rmse -domain imdb  # variant: no neighbors
python3 train.py -ui_variant 0 -orient rmse -domain imdb  # variant: no user_item interaction 
python3 train.py -base_model cnn -orient rmse -domain imdb  # variant: no attention over the tokens
