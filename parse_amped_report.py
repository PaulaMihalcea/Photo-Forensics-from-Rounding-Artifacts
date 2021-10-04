import os
import pandas as pd

# Create report...
if not os.path.isfile('results/report.csv'):

    dir_path = ''  # Path of the Amped Authenticate HTML report goes here

    if dir_path[-1] != '/':
        dir_path += '/'

    file_list = [f for f in os.listdir(dir_path) if f.endswith('html')]
    file_list.remove('report.html')

    dict_df = {'img_name': [], 'dimples_strength': []}

    for file in file_list:
        with open(dir_path + file) as f:
            lines = f.readlines()
            if 'Image shows JPEG Dimples artifacts.' in lines[8].replace('<td><pre>', '').replace('</pre></td>', ''):  # Only get images containing dimples
                if '[0, 0]' in lines[18].replace('<td><pre>', '').replace('</pre></td>', ''):  # Only get images without offset
                    filename = file.split('-')[:-1][0]  # File name
                    dimples_strength = float(lines[13].replace('<td><pre>', '').replace('</pre></td>', ''))  # Dimples strength
                    dict_df['img_name'].append(filename)
                    dict_df['dimples_strength'].append(dimples_strength)

    df = pd.DataFrame.from_dict(dict_df)

    if not os.path.exists('results/'):
        os.makedirs('results/')

    df.to_csv('results/report.csv', index=False)

# ...then load existing report and choose n files for each of three dimples strength
df = pd.read_csv('results/report.csv')
df = df.sort_values('dimples_strength')

df.drop(df[df.dimples_strength < 15].index, inplace=True)

df_lo = df[(df.dimples_strength >= 15) & (df.dimples_strength < 30)]  # Low dimples strength
df_md = df[(df.dimples_strength >= 30) & (df.dimples_strength < 45)]  # Medium dimples strength
df_hi = df[(df.dimples_strength >= 45)]  # High dimples strength

n = 0  # Number of desired images goes here

lo_samples = df_lo.sample(n=n)
md_samples = df_md.sample(n=n)
hi_samples = df_hi.sample(n=n)

print(lo_samples)
print(md_samples)
print(hi_samples)
