import os
import glob
import xml.etree.ElementTree as ET


class XMLtoInputConvertor:
    def __init__(self):
        self.class_names = list()

    def parse_xml(self, img_dir):
        annotations = list()
        for xml_file in glob.glob(img_dir + '/*.xml'):

            tree = ET.parse(open(xml_file))
            root = tree.getroot()
            image_name = root.find('filename').text
            img_path = img_dir + '/' + image_name

            positions = ''

            for i, obj in enumerate(root.iter('object')):
                # difficult = obj.find('difficult').text
                class_name = obj.find('name').text

                if class_name not in self.class_names:
                    self.class_names.append(class_name)

                class_id = self.class_names.index(class_name)
                xmlbox = obj.find('bndbox')

                position = (str(int(float(xmlbox.find('xmin').text))) + ','
                            + str(int(float(xmlbox.find('ymin').text))) + ','
                            + str(int(float(xmlbox.find('xmax').text))) + ','
                            + str(int(float(xmlbox.find('ymax').text))) + ','
                            + str(class_id))

                positions += ' ' + position

            annotations.append(img_path + positions)

        return annotations

    def convert(self, XML_dir: str, annotation_path: str = None):
        annotations = self.parse_xml(XML_dir)
        if annotation_path is not None:
            with open(annotation_path, 'w') as f:
                f.writelines(map(lambda x: x + '\n', annotations))

        return annotations

    def export_class_names(self, class_names_path: str = None):
        class_names = map(lambda x: x + '\n', self.class_names)

        if class_names_path is not None:
            with open(class_names_path, 'w') as f:
                f.writelines(class_names)

        return self.class_names
