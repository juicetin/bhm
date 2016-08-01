import cProfile
import pstats

stat_dir = 'stats/'

def get_stats(stat_file):
    stat_path = stat_dir + stat_file
    p = pstats.Stats(stat_path)
    p.sort_stats('cumtime').print_stats(15)
    return p

stat_file = 'profile_fixed_dK_evals_20points'
p = get_stats(stat_file)
