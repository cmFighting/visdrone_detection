# 写xml

from xml.dom.minidom import *

# 创建一个文档对象
doc = Document()

# 创建一个根节点
root = doc.createElement('managers')

# 根节点添加属性
root.setAttribute('company', '中体彩')
print(root.getAttribute('company'))

# 根节点加入到tree
doc.appendChild(root)

# 创建二级节点
company = doc.createElement('gloryroad')
name = doc.createElement('name')
name.appendChild(doc.createTextNode('公司名称'))  # 添加文本节点

# 创建一个带着文本节点的子节点
ceo = doc.createElement('ceo')
ceo.appendChild(doc.createTextNode('吴总'))  # <ceo>吴总</ceo>

company.appendChild(name)  # name加入到company
company.appendChild(ceo)
root.appendChild(company)  # company加入到根节点

print(ceo.tagName)

print(doc.toxml())

# 存成xml文件
fp = open('files/test.xml', 'w', encoding='utf-8')
doc.writexml(fp, indent='', addindent='\t', newl='\n', encoding='utf-8')
fp.close()
