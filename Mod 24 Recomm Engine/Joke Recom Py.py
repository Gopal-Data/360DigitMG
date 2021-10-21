import pandas as pd
import surprise 
from surprise import SVD
from collections import defaultdict
from surprise.model_selection import train_test_split

#First train an SVD algorithm on the movielens dataset.
data = pd.read_excel ("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 24\\Ratings.xlsx")
data= data.drop(columns=['id'])
data= data.rename(columns={"user_id": "uid", "joke_id": "iid"})

lower_rating = data['Rating'].min()
upper_rating = data['Rating'].max()
print('Review range: {0}to{1}'.format(lower_rating,upper_rating))

reader = surprise.Reader(rating_scale=(-10.0,10.0))
data=surprise.Dataset.load_from_df(data, reader)

def get_top_n(predictions, n=5):
        # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

trainset, testset = train_test_split(data, test_size=.25)

trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)

# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)

top_n = get_top_n(predictions, n=5)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():    
    print(uid, [iid for (iid, _) in user_ratings])