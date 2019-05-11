import os
import glob


def create_txt_file(src):

    main = os.fsencode(src)
    f = open('train_names.txt', 'w')

    for filename in glob.iglob(src + '**/*.JPEG', recursive=True):
        image = os.path.relpath(filename, src)
        f.write(image + '\n')

    f.close()



if __name__ == '__main__':
    create_txt_file('Dataset/Train/images/')