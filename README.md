# Expedia recommendation system

The topic of this assignment is positioned in the area of recommender systems. More specifically, the task is to predict what hotel a user is most likely to book. This could
greatly help companies such as Expedia (from which the dataset actually originates) to organize the search results for a user in the most suitable way.

## 1. Data Collection

The dataset can be downloaded from the Data Mining Techniques class on the Kaggle website (https://www.kaggle.com/c/2nd-assignment-dmt-2021/). It is split into a training and a test set (train.csv and test.csv respectively, each containing approximately 5 million records). Essentially, the dataset contains information about a search query of a user for a hotel, the hotel properties that resulted and for the training set, whether the user clicked on the hotel and booked it. The datasets can also be found in this repository.

### 1.2 Data features

| Field | Data Type | Description |
| --- | --- | --- |
| srch_id | Date/time | Date and time of the search |
| date_time | Integer | ID of the Expedia point of sale (i.e. Expedia.com, Expedia.co.uk, Expedia.co.jp, ..) |
| site_id | Integer | The ID of the search |
| visitor_location_country_id | Integer | The ID of the country the customer is located |
| visitor_hist_starrating | Float | The mean star rating of hotels the customer has previously purchased; null signifies there is no purchase history on the customer |
| visitor_hist_adr_usd | Float | Themean price per night (in US$) of the hotels the customer has previously purchased; null signifies there is no purchase history on the customer |
| prop_country_id | Integer | The ID of the country the hotel is located in |
| prop_id | Integer | The ID of the hotel |
| prop_starrating | Integer |  The star rating of the hotel, from 1 to 5, in increments of 1. A 0 indicates the property has no stars, the star rating is not known or cannot be publicized |
| prop_review_score | Float | The mean customer review score for the hotel on a scale out of 5, rounded to 0.5 increments. A 0 means there have been no reviews, null that the information is not available |
| prop_brand_bool | Integer | +1 if the hotel is part of a major hotel chain; 0 if it is an independent hotel |
| prop_location_score1 | Float | A (first) score outlining the desirability of a hotel’s location |
| prop_location_score2 | Float | A (second) score outlining the desirability of the hotel’s location |
| prop_log_historical_price | Float | The logarithm of themean price of the hotel over the last trading period. A 0 will occur if the hotel was not sold in that period |
| price_usd | Float | Displayed price of the hotel for the given search. Note that different countries have different conventions regarding displaying taxes and fees and the value may be per night or for the whole stay |
| promotion_flag | Integer | +1 if the hotel had a sale price promotion specifically displayed |
| srch_destination_id | Integer | ID of the destination where the hotel search was performed |
| srch_length_of_stay | Integer | Number of nights stay that was searched |
| srch_booking_window | Integer | Number of days in the future the hotel stay started from the search date |
| srch_adults_count | Integer | The number of adults specified in the hotel room |
| srch_children_count | Integer | The number of (extra occupancy) children specified in the hotel room |
| srch_room_count | Integer | Number of hotel rooms specified in the search |
| srch_saturday_night_bool | Boolean | +1 if the stay includes a Saturday night, starts fromThursday with a length of stay is less than or equal to 4 nights (i.e. weekend); otherwise 0 |
| srch_query_affinity_score | Float | The log of the probability a hotel will be clicked on in Internet searches (hence the values are negative) A null signifies there are no data (i.e. hotel did not register in any searches) |
| orig_destination_distance | Float | Physical distance between the hotel and the customer at the time of search. A null means the distance could not be calculated |
| random_bool | Boolean | +1 when the displayed sort was random, 0 when the normal sort order was displayed |
| comp1_rate | Integer | +1 if Expedia has a lower price than competitor 1 for the hotel; 0 if the same; -1 if Expedia’s price is higher than competitor 1; null signifies there is no competitive data |
| comp1_inv | Integer | +1 if competitor 1 does not have availability in the hotel; 0 if both Expedia and competitor 1 have availability; null signifies there is no competitive data |
| comp1_rate_percent_diff | Float | The absolute percentage difference (if one exists) between Expedia and competitor 1’s price (Expedia’s price the denominator); null signifies there is no competitive data |
| comp2_rate | Integer | |
| comp2_inv | Integer | |
| comp2_rate_percent_diff | Float | |
| ... |  | |
| comp8_rate | Integer | |
| comp8_in | Integer | |
| comp8_rate_percent_diff | Float | |
|  | |  |
| **Training set only** |
| position | Integer | Hotel position on Expedia’s search results page. This is only provided for the training data, but not the test data |
| click_bool | Boolean | 1 if the user clicked on the property, 0 if not |
| booking_bool | Boolean | 1 if the user booked the property, 0 if not |
| gross_booking_usd | Float | Total value of the transaction. This can differ from the price_usd due to taxes, fees, conventions on multiple day bookings and purchase of a room type other than the one shown in the search |

## 2. Feature engineering


**1. Prediction value**
For the machine learning techniques used in this paper, a prediction value is needed.  Assign score is created with the features 'click\_bool and 'book\_bool'. The score is determined based on the following formula:

Assign_score = 4 * booking_bool + 1 * click_bool

**2. Ranking** 
As the problem dealt with is a ranking problem, the model could be improved by adding the rank of numerical ordinal columns with respect to one specific search. If values of the features are equal, they also receive the same rank. This resulted in four additional features: price rank, star rating rank, review score rank and location score rank.

**3. Competitor rate**
The 3 competitor features are merged together in the columns 'competitor_rate', 'competitor_inv' and 'competitor_rpd'. The last feature is summed and then divided by 100 and 8 to make it an absolute percentage difference again.  

**4. Datetime**
The datetime feature is extracted in the features hour, day, month, quarter and day of the week. These columns were used as extra numerical features. 

**5. Rating and price difference**
Two extra features were created based on a visitors previous booking. The rating includes the difference between the mean star rating of hotels the customer has previously purchased and the current star rating of the hotel.

**6. Clusters**
When all new features were created, the data is divided into 100 clusters. For this paper, the Minibatch K-means clustering algorithm is used.  

## 3. Methodology

### 3.1 LightGBM method

The parameters used for the LightGBM method are a 'gbdt' boosting type with 300 estimators and 50 leaves. 

### 3.2 XGBoost method

The parameters used for the XGBoost method is the 'gbtree' with 300 estimators and 4 jobs. 

## 4. Testing on test data

Both methods were used on the test data set and a table is returned with the recommended hotels per user_id. This table is uploaded in the Kaggle competition and with this the evaluation is calculated. The evaluation metric for this competition is Normalized Discounted Cumulative Gain (NDCG)@5 calculated per query and averaged over all queries with the the values weighted by the log_2 function.

A score of 0.38708 is achieved with this code. The final position on the leaderboard was 41/279.

