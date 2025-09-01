from halib import *
from halib.filetype import textfile

def main():
    indir = r"E:\NextCloud\paper3\datasets\DFire"
    testfile = os.path.join(indir, "video_test.txt")
    lines = textfile.read_line_by_line(testfile)
    pprint(lines)
    test_dir = os.path.join(indir, "test")
    for fname in lines:
        src = os.path.join(indir, fname)
        assert os.path.exists(src), f"Source file {src} does not exist."
        dst = os.path.join(test_dir, fname)
        fs.move_dir_or_file(src, dst)
        print(f"Copied {src} to {dst}")


if __name__ == "__main__":
    main()
