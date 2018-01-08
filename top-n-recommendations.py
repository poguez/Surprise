import os, random
from collections import defaultdict
from surprise import Reader
from surprise import accuracy
from surprise import SVD, GridSearch, BaselineOnly, NormalPredictor, SlopeOne, CoClustering, SVDpp, SVDppMultipleImplicitFeedback
from surprise import NMF, KNNBasic, KNNWithMeans, KNNBaseline, NMF
from surprise import Dataset
from surprise import evaluate, print_perf

binarized_columns = ["author_recipes_count_more_than_average"]\
    # ,
    #                  "actions_binded_to_trigger_more_than_average",
    #                  "action_uses_count_more_than_average",
    #                  "trigger_uses_count_more_than_average",
    #                  "triggers_binded_to_actions_more_than_average"]



def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n



for column in binarized_columns:

    # Dataset load file
    file_path ='/Users/poguez/Desktop/ifttt-recipes/data/'+str(column)+'/ifttt_dataset_for_recommendation_composed_ints.csv'
    reader = Reader(line_format="user item rating timestamp implicit_feedback_2", sep='\t', skip_lines=0)
    data_ifttt_dataset_for_recommendation_composed = Dataset.load_from_file(file_path, reader=reader)
    trainset = data_ifttt_dataset_for_recommendation_composed.build_full_trainset()

    # # First train an SVD algorithm on the movielens dataset.
    # data = Dataset.load_builtin('ml-100k')
    # trainset = data.build_full_trainset()
    algo = SVD()
    algo.train(trainset)

    # # Than predict ratings for all pairs (u, i) that are NOT in the training set.
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)

    top_n = get_top_n(predictions, n=10)

    # Print the recommended items for each user
    for uid, user_ratings in top_n.items():
        print(uid, [iid for (iid, _) in user_ratings])


#
#
# for column in binarized_columns:
#
#     # Dataset load file
#     file_path ='/Users/poguez/Desktop/ifttt-recipes/data/'+str(column)+'/ifttt_dataset_for_recommendation_composed_ints.csv'
#     reader = Reader(line_format="user item rating timestamp implicit_feedback_2", sep='\t', skip_lines=0)
#     data_ifttt_dataset_for_recommendation_composed = Dataset.load_from_file(file_path, reader=reader)
#     data_ifttt_dataset_for_recommendation_composed.build_full_trainset()
#     data_ifttt_dataset_for_recommendation_composed.split(n_folds=10)
#
#     # # We'll use the famous SVD++ algorithm but improved by for TAP.
#     svdPredictor = SVDppMultipleImplicitFeedback(verbose=True)
#
#     data2 = data_ifttt_dataset_for_recommendation_composed
#     print("#################################################\n\n\n " + "Analyzing: "+ column+"\n\n\n#################################################\n\n\n " )
#
#     perf = evaluate(svdPredictor, data2, measures=['RMSE', 'MAE'])
#
#     # print_perf(perf)
#
