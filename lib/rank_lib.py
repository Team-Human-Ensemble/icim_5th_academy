import numpy as np
import pandas as pd

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest


from skfeature.function.similarity_based import fisher_score
from skfeature.utility import construct_W
from skfeature.function.similarity_based import lap_score


def get_feature_ranking(raw_data,chi_rank, fisher_rank,lap_rank):
    '''
    Input
        chi_rank : chi_squared ranking 
        fisher_rank : fisher ranking
        lap_rank : laplacian ranking
    Output
        final_rank : final ranking, chi + fisher + lap, sorting result
        
    '''
    feature_len = np.shape(raw_data)[1]
    
    final_rank = np.zeros(feature_len)
    
    col_name_list = raw_data.columns
    df_feature = pd.DataFrame(col_name_list)

    reverse_chi = np.flip(chi_rank,0)
    reverse_fisher = np.flip(fisher_rank,0)
    reverse_lap = np.flip(lap_rank,0)
    
    chi_rank_score_list = []
    fisher_rank_score_list = []
    lap_rank_score_list = []
    
    
    for idx in range(len(reverse_chi)):
        rank_score = (1/29) * idx 
        chi_rank_score_list.append((reverse_chi[idx],rank_score))
        fisher_rank_score_list.append((reverse_fisher[idx],rank_score))
        lap_rank_score_list.append((reverse_lap[idx],rank_score))
        
        final_rank[reverse_chi[idx]] = final_rank[reverse_chi[idx]] + rank_score
        final_rank[reverse_fisher[idx]] = final_rank[reverse_fisher[idx]] + rank_score
        final_rank[reverse_lap[idx]] = final_rank[reverse_lap[idx]] + rank_score

    #print(final_rank)
    #print(chi_rank_score_list)
    
    df_ranking = pd.DataFrame(final_rank)
    df_col = pd.DataFrame(col_name_list)
    featureScores = pd.concat([df_col,df_ranking],axis=1)
    featureScores.columns = ['Feature','ranking']  #naming the dataframe columns
    
    
    result = featureScores.nlargest(30,'ranking')
    print(result)
    
    return final_rank, result


def get_fisher_score(data,label,k = 30):
    score = fisher_score.fisher_score(data, label)
    #print(score)
    ranking = fisher_score.feature_ranking(score)
    #print(idx)
    
    
    dfscores = pd.DataFrame(score)
    dfcolumns = pd.DataFrame(data.columns)
    #df_rank =pd.DataFrame(idx)
    
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Feature','Score']  #naming the dataframe columns
    #print(featureScores.nlargest(k,'Score'))  #print 20 best features
    result = featureScores.nlargest(k,'Score')
    
    return result, ranking
    
    
def get_lap_score(data, k=5, t=1,top_feature = 30):
    kwargs_W = {"metric":"euclidean","neighbor_mode":"knn","weight_mode":"heat_kernel","k":k,'t':t}
    W = construct_W.construct_W(data, **kwargs_W)
    score = lap_score.lap_score(data, W=W)
    #print(score)
    ranking = lap_score.feature_ranking(score)
    #print(idx)
    
    dfscores = pd.DataFrame(score)
    dfcolumns = pd.DataFrame(data.columns)
    #df_rank = pd.DataFrame(idx)
    
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Feature','Score']  #naming the dataframe columns
    #print(featureScores.nlargest(k,'Score'))  #print 20 best features
    result = featureScores.nlargest(top_feature,'Score')
    
    return result, ranking
    
    
def get_chi_score(data,label, k =30):
    '''
    Input
        data : pandas dataframe 
        label : binary class label
        k : number of features
        
    Output : sorted kai-score list
    
    '''
    ranking_list = []
    x = data
    y = label 
    
    
    #target column i.e price range
    #apply SelectKBest class to extract top 20 best features
    bestfeatures = SelectKBest(score_func=chi2, k=k)
    fit = bestfeatures.fit(data,label)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(data.columns)
    
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Feature','Score']  #naming the dataframe columns
    #print(featureScores.nlargest(k,'Score'))  #print 20 best features
    result = featureScores.nlargest(k,'Score')
    #print(temp)
    #print(temp.iloc[0])
    ranking_list = list(result['Feature'].index)   
    
    return result,ranking_list

def get_in_cluster_ranking(rank_info_list,cluster_list):
    '''
    choice the feature in cluster lists using ranking system
    
    Input
        rank_info_list
        cluster_list : list of clustering
    Output 
        after_rank_output : feature list
    '''
    after_rank_output = []
    
    for temp_list in cluster_list:
        
        if len(temp_list)>1:
            idx = 0
            
            
            for feature_name in temp_list:
                if idx ==0:
                    prev_feature = rank_info_list.loc[rank_info_list.Feature == feature_name]
                    idx = idx+1
                else:
                    cur_feature = rank_info_list.loc[rank_info_list.Feature == feature_name]
                    if int(prev_feature.ranking) > int(cur_feature.ranking):
                        out_feature = prev_feature
                    else:
                        out_feature = cur_feature
                        
                    prev_feature = out_feature
                    idx = idx+1
            
            after_rank_output.append(out_feature.Feature.iloc[0])
        else:
            after_rank_output.append(temp_list[0])
            
    return after_rank_output