from halib import *
from video_db.archived.hrwen_provider import HpwrenLbProvider
from video_db.newdb_provider import NewDBLbProvider


def main():
    # label_provider = HpwrenLbProvider()
    # label_provider.process_labeling(to_csv=True)
    new_label_provider = NewDBLbProvider()
    new_label_provider.process_labeling(to_csv=True, max_workers=0) # no threading

if __name__ == "__main__":
    main()
