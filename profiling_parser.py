import cProfile
import pstats

stat_dir = 'stats/'
stat_file = 'profile_vectorized_sqeucl_dist'
stat_path = stat_dir + stat_file
p = pstats.Stats(stat_path)
p.print_stats()
