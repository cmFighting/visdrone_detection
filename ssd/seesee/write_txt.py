'''
用于制作train val 和test文本文件，通过遍历的形式

'''
import os


def files2txt(folder_path, txt_save_path):
    # txt_annotations_path = 'F:\\datas\\VOC\\DET\\VisDrone2019-DET-train\\annotations_txt\\'
    names = os.listdir(folder_path)
    with open(txt_save_path, 'w') as f:
        for name in names:
            x = name.strip('.xml')
            f.writelines(x)
            f.write('\n')


if __name__ == '__main__':
    # folder_path = 'F:\\datas\\VOC\\VisDrone_ROOT\\VisDrone2019\\annotations_xml'
    # folder_path = 'F:\\datas\\VOC\\DET\\VisDrone2019-DET-val\\annotations_xml'
    folder_path = 'F:\\datas\\VOC\\DET\\VisDrone2019-DET-test_dev\\annotations_xml'
    txt_save_path = 'Main/test.txt'

    files2txt(folder_path, txt_save_path)


    # file_path = "te.txt"
    # mylist = ["100", "200", "300"]
    # file_write_obj = open(file_path, 'w')  # 以写的方式打开文件，如果文件不存在，就会自动创建
    # for var in mylist:
    #     file_write_obj.writelines(var)
    #     file_write_obj.write('\n')
    # file_write_obj.close()
