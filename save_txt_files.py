import os


def create_txt_file(src):

    main = os.fsencode(src)
    f = open('valid_names.txt', 'w')

    for folder in os.listdir(main):
        foldername = src + "/" + os.fsdecode(folder)
        for file in os.listdir(os.fsencode(foldername)):
            filename = os.fsdecode(file)
            if filename.endswith(('.JPEG', '.png', '.jpg')) and not filename.startswith("."):  # image extension we need
                image = filename
                f.write(image + '\n')
    f.close()



if __name__ == '__main__':
    create_txt_file('Dataset/Validation')