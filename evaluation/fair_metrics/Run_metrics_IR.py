import pandas as pd
import numpy as np

import bookgender.fair_metrics.singh_joachims as sj
import bookgender.fair_metrics.sapiezynski as sp
import bookgender.fair_metrics.biega as bg
import bookgender.fair_metrics.diaz as dz
import bookgender.fair_metrics.zehlike as zh

import bookgender.metric_utils.groupinfo as gi
import bookgender.metric_utils.position as pos
#import bookgender.metrics.load_goodreads as lg


class metric_analysis:
    
    ranked_list = None
    test_rates = None
    group = None
    arg = None
    arg_val = None
    
    def __init__(self, ranked_list, test_rates, group, original_data=None, IAA=True, EE=True, AWRF=True, DRR=True, FAIR=True):
        
        self.ranked_list = ranked_list
        self.test_rates = test_rates
        self.original_data = original_data
        self.group = group
        self.IAA = IAA
        self.EE= EE
        self.DRR = DRR
        self.FAIR = FAIR
        self.AWRF=AWRF
        
    def run_IAA(self, user_data, pweight=pos.geometric()):
        
        
        """
        Measure fairness using IAA metric.
        Args:
            ranked_list(panda.DataFrame): truncated ranked list of retrieved results
            pweight(position object): user browsing model to measure position weight. Default: geometric
        Return:
            pandas.Series: fairness score of ranked lists for each algorithms
        """
        
        weight_vector = pweight(ranked_list)
        algo = ranked_list['system'].unique()
        orb = self.original_data.loc[self.original_data['system']==algo[0]]
        return pd.Series({'IAA': bg.unfairness(ranked_list, self.original_data, self.group, weight_vector)})
    
    def run_EE(self, ranked_list, pweight=pos.cascade()):
        
        """
        Measure fairness using expected-exposure metric.
        Args:
            ranked_list(panda.DataFrame): truncated ranked lists of retrieved results
            pweight(position object): user browsing model to measure position weight. Default: cascade
        Return:
            pandas.Series: fairness score of ranked lists for each algorithms
        """
        
        return dz.ee_for_ds(ranked_list, self.test_rates, self.group, pweight)
 
    def run_dp_eur_rur(self, ranked_list, pweight=pos.logarithmic()):
        
        """
        Measure fairness using logDP, logEUR, and logRUR metrics.
        Args:
            ranked_list(panda.DataFrame): truncated ranked lists of retrieved results
            pweight(position object): user browsing model to measure position weight. Default: logarithmic
        Return:
            pandas.Series: fairness score of ranked lists for each algorithms
        """
        
        weight_vector = pweight(user_data)
        return pd.Series({
            'logDP': sj.demographic_parity(ranked_list, self.group, weight_vector),
            'logEUR': sj.exposed_utility_ratio(ranked_list, self.test_rates, self.group, weight_vector),
            'logRUR': sj.realized_utility_ratio(ranked_list, self.test_rates, self.group, weight_vector),

        })
    def run_FAIR(self, ranked_list):
        
        """
        Measure fairness using FAIR metric.
        Args:
            ranked_list(panda.DataFrame): truncated ranked lists of retrieved results
            
        Return:
            pandas.Series: fairness score of ranked lists for each algorithms
        """
        return pd.Series(zh.avg_prefix(ranked_list, self.group))
        
    
    def run_awrf(self, ranked_list, pweight):
        
        """
        Measure fairness using AWRF metric.
        Args:
            ranked_list(panda.DataFrame): truncated ranked lists of retrieved results.
            pweight(position object): user browsing model to measure position weight.
        Return:
            pandas.Series: fairness score of ranked lists for each algorithms
        """
        
        weight_vector = pweight(ranked_list)
        user_awrf = pd.Series({'AWRF': sp.awrf(ranked_list, self.group, weight_vector)})
        return user_awrf
    
    def run_awrf_fair(self, ranked_list, pweight=pos.geometric()):
        
        """
        Measure fairness using single ranking metrics.
        Args:
            ranked_list(panda.DataFrame): truncated ranked lists of retrieved results.
            pweight(position object): user browsing model to measure position weight. Default: geomtric
        Return:
            pandas.Series: fairness scores of ranked lists for each algorithms
        """
        
        weight_vector = pweight(ranked_list)
        return pd.Series({
            'AWRF_equal': sp.awrf(ranked_list, self.group, weight_vector, p_hat=0.5),
            'FAIR': zh.avg_prefix(ranked_list, self.group)
        }).append(self.run_awrf(ranked_list, pweight))
    
    def run_stochastic_metric(self, ranked_list, pweight):
        
        """
        Measure fairness using single ranking metrics.
        Args:
            ranked_list(panda.DataFrame): truncated ranked lists of retrieved results.
            pweight(position object): user browsing model to measure position weight.
        Return:
            pandas.Series: fairness scores of ranked lists for each algorithms
        """
        
        result = pd.Series()
        if pweight == 'default':
            if self.IAA == True:
                result = self.run_IAA(ranked_list)
            if self.EE == True:
                result = result.append(self.run_EE(ranked_list))
            if self.DRR == True:
                result = result.append(self.run_dp_eur_rur(ranked_list))
            return result
        if self.IAA == True:
            result = self.run_IAA(ranked_list, pweight)
        if self.EE == True:
            result = result.append(self.run_EE(ranked_list, pweight))
        if self.DRR == True:
            result = result.append(self.run_dp_eur_rur(ranked_list, pweight))
        
        return result
      
    def run_sensitivity_analysis(self, position_weight, arg=None, arg_val=None, listsize=10):
        
        truncated = self.ranked_list[self.ranked_list['rank']<=listsize]
        #truncated = self.ranked_list
        
        if arg == 'stop':
            pweight = position_weight(stop=arg_val)
        elif arg == 'patience':
            pweight = position_weight(patience=arg_val)
        else:
            pweight = position_weight()
            
        stochastic_metric = truncated.groupby(['system', 'qid']).progress_apply(self.run_stochastic_metric, pweight=pweight)
        stochastic_metric_mean = stochastic_metric.groupby('system').mean()
        stochastic_metrics_score = stochastic_metric_mean.reset_index().melt(id_vars=['system'], var_name='Metric')
        #print(other_metrics_score)
        
        if self.AWRF == True:
            truncated = truncated[truncated['sequence'].str.endswith('.0')]
            user_awrf = truncated.groupby(['system', 'qid']).progress_apply(self.run_awrf, pweight = pweight)
            user_agg = user_awrf.groupby(['system']).mean()
            AWRF = user_agg.reset_index().melt(id_vars=['system'], var_name='Metric')
            final_metric = pd.concat([AWRF, stochastic_metrics_score], ignore_index=True)
            
        else:
            final_metric = stochastic_metrics_score
        
        final_metric[arg] = arg_val
        final_metric['pos_weight'] = str(position_weight).split('.')[-1][:-2]
        final_metric['ranked_size'] = listsize
        
        return final_metric
    
    def run_default_setting(self, listsize):
        
        truncated = self.ranked_list[self.ranked_list['rank']<=listsize]
        #truncated = self.ranked_list
        
        stochastic_metrics = truncated.groupby(['system', 'qid']).progress_apply(self.run_stochastic_metric, pweight='default')
        stochastic_metrics_mean = stochastic_metrics.groupby('system').mean()
        stochastic_metrics_score = stochastic_metrics_mean.reset_index().melt(id_vars=['system'], var_name='Metric')
        
        if self.AWRF == True or self.FAIR == True:
            user_awrf_fair = truncated.groupby(['system', 'qid']).progress_apply(self.run_awrf_fair)
            user_agg = user_awrf_fair.groupby(['system']).mean()
            AWRF_FAIR = user_agg.reset_index().melt(id_vars=['system'], var_name='Metric')
            final_metric = pd.concat([AWRF_FAIR, stochastic_metrics_score], ignore_index=True)
        else:
            final_metric = stochastic_metrics_score
            
        final_metric['ranked_size'] = listsize
        
        return final_metric
        
        
        
        

    
    
    
    
    
        
        
        