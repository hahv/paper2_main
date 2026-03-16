import pandas as pd
from io import StringIO
csv_data = """ ; ; ;gt_label;gt_label;temp_method_motion_block;temp_method_motion_block;temp_method_motion_block;temp_method_motion_block; 
VIDEO NAME;VIDEO_PATH;FRAMES;FireSmoke;None;Correct Infer.;Correct Skip;False Skip;Wasted Infer.;VISUALIZATION
TOTAL;TOTAL;1800;60.0000% (1080);40.0000% (720);98.3333% (1062);48.7500% (351);1.6667% (18);51.2500% (369);-"""
df = pd.read_csv(StringIO(csv_data), sep=';', header=[0, 1])
target_col = None
for col in df.columns:
    if str(col[0]).strip().startswith('temp_method_') and str(col[1]).strip() == 'Correct Skip':
        target_col = col
        break
print(target_col)
first_col = df.columns[0]
print(first_col)
total_row = df[df[first_col].astype(str).str.strip() == 'TOTAL']
print(total_row[target_col].iloc[0] if not total_row.empty else "No")
