# port app 

input details
================
**expected input* : item name , port name, vessel name

*optional input* : quantity, quantity unit, vendor name, purchase type


*quantity *: if not given , take last given value

*quantity unit*: if not given take the last known value

*purchase type:* take the most frequest value for that item 

*vendor name *: if not given take the first value from same item-port data

Process
=========
###### Endpoints ######
1) /health : for health check of application 
2) /train: For training and evalation 

    a) Take the whole data 

    b) Preprocess the data split in to strong and week dataset

    c) Clean the dataset

    d) Train and test split : 80% in to training and 20 in to testing after sorting by year , month, day wise

    e) Based on item_port segment apply the model 

    f) Evaluate the model and save combined metrics: R2, RMSE, MSE


3) /retrain:  For model retraining : Take the data so far available from db , retrain on same model and save the artifacts and model details in local file folder . 

    2 buckets : item port value count ( greater than 10, less than 10 )

    Models used: small dataset - Ridge    large dataset -  RandomForestRegressor

    model version is saved in the format YerMonth folder inside models

    'port_segment_map.pkl' - saved the below details in this file

    {'strong_ports': strong_ports,
        'weak_ports': weak_ports,
        'strong_threshold': strong_threshold,
        'item_port_avg_price': avg_prices_combined 
    } 

    item_port_avg_price: monthly avg price of item_port combinations

    |____________________ Root folder

        |________________ models
            |____________ high_freq_YearMonth
                |________ encoder.pkl
                |________ model.pkl
            |____________ low_freqYearMONTH
                |________ encoder.pkl
                |________ model.pkl
        |________________ port_segment_map.pkl



    encoder: Categorical value encodingg mapping 
    model: saved model
    current threshld: 10 data points on item-port combinations



4) /predict: inference pipeline input details are given at the top 

    a) Get the input details

    b) Find out item_port se fgmentrom saved port_segment

    c) Select suitable model based on port-item segment, take the previous mnth model saved and apply on datapoint

    d) Apply the model 

    c) Pass the result and save in db: 
        vESSEL ID, PORT NAME, VENDOR NAME , PURCHASE TYPE, ITEM_UNIT_CATEGORY, QUANTITY, DATE OF PREDICTION , PREDICTED UNIT PRICE IN USD , MODEL NAME


    