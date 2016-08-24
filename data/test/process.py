import numpy as np

def char_m_to_str_v(m):
    return np.array([''.join(char_arr) for char_arr in m])

with open('ILEA567.DAT') as f:
    raw = f.readlines()
    # for idx, line in enumerate(f.readlines()):
    #     line = line.strip()
    #     if len(line) != 16:
    #         print(line)
    #         print(idx)

raw = np.array([list(string) for string in raw])
raw = raw[:,0:16]

years = raw[:,0].astype(int)
schools = char_m_to_str_v(raw[:,1:4]).astype(int)
scores = char_m_to_str_v(raw[:,4:6]).astype(int)
fsm = char_m_to_str_v(raw[:,6:8]).astype(int)
vr1bandperc = char_m_to_str_v(raw[:,8:10]).astype(int)
gender = raw[:,10].astype(int)
vrband = raw[:,11].astype(int)
ethnicity = char_m_to_str_v(raw[:,12:14]).astype(int)
schoolgender = raw[:,14].astype(int)
schooldenom = raw[:,15].astype(int)
