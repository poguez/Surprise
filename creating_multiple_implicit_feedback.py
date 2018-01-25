import os, random
from surprise import Reader
from surprise import accuracy
from surprise import SVD, GridSearch, BaselineOnly, NormalPredictor, SlopeOne, CoClustering, SVDpp, SVDppMultipleImplicitFeedback
from surprise import NMF, KNNBasic, KNNWithMeans, KNNBaseline, NMF
from surprise import Dataset
from surprise import evaluate, print_perf



binarized_columns = [
                    "author_recipes_count_more_than_average",
                    "actions_binded_to_trigger_more_than_average",
                     "action_uses_count_more_than_average",
                    "trigger_uses_count_more_than_average",
                     "triggers_binded_to_actions_more_than_average"]

for column in binarized_columns:

    # Dataset load file
    file_path ='/Users/poguez/Desktop/ifttt-recipes/data/'+str(column)+'/ifttt_dataset_for_recommendation_composed_ints.csv'
    reader = Reader(line_format="user item rating timestamp implicit_feedback_2", sep='\t', skip_lines=0)
    data_ifttt_dataset_for_recommendation_composed = Dataset.load_from_file(file_path, reader=reader)
    data_ifttt_dataset_for_recommendation_composed.split(n_folds=2)


    # We'll use the famous SVD++ algorithm but improved by for TAP.
    svdPredictor = SVDppMultipleImplicitFeedback(verbose=True)

    data2 = data_ifttt_dataset_for_recommendation_composed
    print("#################################################\n\n\n " + "Analyzing: "+ column +"\n\n\n#################################################\n\n\n " )

    perf = evaluate(svdPredictor, data2, measures=['RMSE', 'MAE'])

    # print_perf(perf)
