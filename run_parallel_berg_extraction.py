from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed

from datetime import datetime
from nwb_extraction import generated_formatted_features_output


start_time = datetime.now()
gouwens_path = Path('/external/rprshnas01/netdata_kcni/stlab/Public/AIBS_patchseq_2020/mouse/ephys/000020')
gouwens_nwbs = gouwens_path.glob('*.nwb')


if __name__ == '__main__':
    print(f'Starting processing at: {start_time}')
    result = Parallel(n_jobs=12)(delayed(generated_formatted_features_output)(str(nwb)) for nwb in gouwens_nwbs)
    print(f"Processing finished at {datetime.now()} ")
    print(f"Total time taken {datetime.now() - start_time} ")
    feature_df = pd.DataFrame(result)
    feature_df.set_index('filename')
    feature_df.to_csv('human_ephys_features_gouwens_w_QC.csv')