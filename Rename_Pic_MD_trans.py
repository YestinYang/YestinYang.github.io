import os

dir = "d:/new post"
blog_date = "2018-01-29"

def rename_file(dir, blog_date):
    # 1) get the list of file name
    file_list = os.listdir(dir)  # r for do not modify the result
    file_list_png = [i for i in file_list if ".md" not in i]  # get only picture files

    # 1.5) handle working directory
    saved_dir = os.getcwd()
    os.chdir(dir)
    print ("Working directory is changed to " + dir)

    # 2) rename them
    for file_name in file_list_png:
        os.rename(file_name, blog_date+"_"+file_name)

    # 2.5) back to original dir
    print ("Working directory is changed back to " + saved_dir)
    os.chdir(saved_dir)

rename_file(dir, blog_date)
