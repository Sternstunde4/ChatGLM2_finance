# 文档切分
import re
import os
import time
import transformers
import dashscope
from dashscope import TextEmbedding

from dashvector import Client, Doc

from Segmentation.generator import AutoIncrementIDGenerator

# 所有可能的细分标题
titles_1 = ['一、', '二、', '三、', '四、', '五、', '六、', '七、', '八、', '九、', '十、',
            '十一、', '十二、', '十三、', '十四、', '十五、', '十六、', '十七、', '十八、', '十九、', '二十、']
titles_2 = ['（一）', '（二）', '（三）', '（四）', '（五）', '（六）', '（七）', '（八）', '（九）', '（十）',
            '（十一）', '（十二）', '（十三）', '（十四）', '（十五）', '（十六）', '（十七）', '（十八）', '（十九）', '（二十）']
titles_3 = ['1、', '2、', '3、', '4、', '5、', '6、', '7、', '8、', '9、', '10、',
            '11、', '12、', '13、', '14、', '15、', '16、', '17、', '18、', '19、', '20、']
titles_4 = ['（1）', '（2）', '（3）', '（4）', '（5）', '（6）', '（7）', '（8）', '（9）', '（10）',
            '（11）', '（12）', '（13）', '（14）', '（15）', '（16）', '（17）', '（18）', '（19）', '（20）']
titles_5 = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩']
titles_6 = ['1）', '2）', '3）', '4）', '5）', '6）', '7）', '8）', '9）', '10）',
            '11）', '12）', '13）', '14）', '15）', '16）', '17）', '18）', '19）', '20）',]


# 获取某个页码范围内的内容
def get_content_on_page(toc_entries, start_page,end_page):
    page_content = ""
    for toc_entry in toc_entries:
        if (int(toc_entry[0]) >= start_page) & (int(toc_entry[0]) <= end_page):

            page_content += toc_entry[3] + "\n"
        elif int(toc_entry[0]) > end_page:
            break

    return page_content


# 获取目录页的章节标题
def process_content_page(toc_entries):
    page_content = ""
    for toc_entry in toc_entries:
        if toc_entry[3].endswith('目录'):
            page_number = int(toc_entry[0])
            page_content = get_content_on_page(toc_entries, page_number, page_number)
            break
    sections_and_pages = gen_section_titles(page_content)
    return sections_and_pages


# 将文本切分为元组（含有页码和行数）
def txt2toc(file_path):
    text = open(file_path, 'r', encoding='utf-8').read()
    toc_pattern = r"\{'page':\s*<Page:(?P<page>\d+)>,\s*'allrow':\s*(?P<allrow>\d+),\s*'type':\s*'(?P<type>\w+)',\s*'inside':\s*(?:\"(?P<inside>[^\"]+)\"|'(?P<inside_simple>[^']+)')\}"
    matches2 = re.findall(toc_pattern, text, re.MULTILINE)
    toc_entries = []
    for match2 in matches2:
        page, allrow, type_value = match2[0], match2[1], match2[2]
        inside = match2[3]
        inside_simple = match2[4]
        # 选择inside或inside_simple中的一个非空值
        if inside:
            toc_entries.append((page, allrow, type_value, inside))
        elif inside_simple:
            toc_entries.append((page, allrow, type_value, inside_simple))
    return toc_entries


# 获取目录中的章节标题
def gen_section_titles(text):
    section_titles = []
    pattern = r"(第\S+[节章])*(.*?)\.{5,}.*\d*"
    matches1 = re.findall(pattern, text, re.MULTILINE)
    for match1 in matches1:
        complete_match1 = match1[0] + match1[1]
        section_titles.append(complete_match1)
    return section_titles


# 根据章节标题的list来切分text
def divide_sections(text, section_titles):
    all_sections = {}
    for i, section_title in enumerate(section_titles):
        try:
            if i < len(section_titles) - 1:
                section_content = text.split(section_title + "\n")[1].split(section_titles[i + 1] + "\n")[0]
            else:
                section_content = text.split(section_title + "\n")[1]
            all_sections[section_title] = section_content
        except:
            pass
    return all_sections


# 替换表格
def replace_tables(parsed_table, new_begin):
    headers = parsed_table[new_begin]
    col_headers = headers[1:]
    extracted_info = []
    for j, row in enumerate(parsed_table[new_begin + 1:]):
        if len(row) != len(headers):  # 另一个表格
            break
        else:
            row_header = row[0]
            for i, col_header in enumerate(col_headers):
                value = row[i + 1]
                extracted_info.append(f'{row_header}:{col_header}:{value}')
        new_begin += 1
    new_begin += 1
    new_tables = '\n'.join(extracted_info)
    return new_tables, new_begin


# 处理表格,行标放前面，紧跟列标，然后是表格中的值
def process_table(content):
    table = re.findall(r'\[[^[\]]*?\](?:\s*,\s*\[[^[\]]*?\])*\n?', content)  # r'\[\s*\'.*?\[.*?\].*?|.*?\',\s*\'.*?\[.*?\].*?|.*?\'*\s*\]\n'
    parsed_table = [eval(tbl) for tbl in table]
    if len(parsed_table) != 0:
        new_begin = 0
        while (new_begin < len(parsed_table)):
            new_tables, new_begin_1 = replace_tables(parsed_table, new_begin)
            old_tables = ''
            for i in range(new_begin, new_begin_1):
                # 替换原来的表格
                old_tables = old_tables + '\n' + table[i]
            content = content.replace(old_tables, new_tables)
            new_begin = new_begin_1
    return content


# 限制切分的最大长度
# def max_length_divide(input_text, max_length):
#     if len(input_text) <= max_length:
#         return input_text, ""
#     truncated_text = input_text[:max_length]
#     last_period_index = truncated_text.rfind("。")
#
#     if last_period_index != -1:
#         extracted_text = truncated_text[:last_period_index + 1]
#         remaining_text = input_text[last_period_index + 1:]
#     else:
#         extracted_text = ""
#         remaining_text = input_text[max_length:]
#     return extracted_text, remaining_text


def mask_entity(entity,content):
    new_content = content.replace(entity[0], "公司")  # 公司全称
    new_content = new_content.replace(entity[1], "公司")  # 公司简称
    return new_content


# 对content进行所有可能的细分标题切分，得到最小粒度的内容
def segmentation(entity, file_name, content, parent, id_generator, type = 0):

    pattern = rf'\n({"|".join(titles_1)})(.*?)\n'
    matches4 = re.findall(pattern, content, re.MULTILINE)
    pattern = rf'\n({"|".join(titles_2)})(.*?)\n'
    matches5 = re.findall(pattern, content, re.MULTILINE)
    pattern = rf'\n({"|".join(titles_3)})(.*?)\n'
    matches6 = re.findall(pattern, content, re.MULTILINE)
    pattern = rf'\n({"|".join(titles_4)})(.*?)\n'
    matches7 = re.findall(pattern, content, re.MULTILINE)
    pattern = rf'\n({"|".join(titles_5)})(.*?)\n'
    matches8 = re.findall(pattern, content, re.MULTILINE)
    pattern = rf'\n({"|".join(titles_6)})(.*?)\n'
    matches9 = re.findall(pattern, content, re.MULTILINE)
    if len(matches4) > 0 & type != 1:
        complete_matches4 = []
        for match4 in matches4:
            complete_match4 = match4[0] + match4[1]
            complete_matches4.append(complete_match4)
        sections_1 = divide_sections(content, complete_matches4)
        for section, content in sections_1.items():
            private_parent = parent + ' 中的 ' + section
            content = '\n' + content
            segmentation(entity, file_name, content, private_parent, id_generator, 1)
    elif len(matches5) > 0 & type != 2:
        complete_matches5 = []
        for match5 in matches5:
            complete_match5 = match5[0] + match5[1]
            complete_matches5.append(complete_match5)
        sections_1 = divide_sections(content, complete_matches5)
        for section, content in sections_1.items():
            private_parent = parent + ' 中的 ' + section
            content = '\n' + content
            segmentation(entity, file_name, content, private_parent, id_generator, 2)
    elif len(matches6) > 0 & type != 3:
        complete_matches6 = []
        for match6 in matches6:
            complete_match6 = match6[0] + match6[1]
            complete_matches6.append(complete_match6)
        sections_1 = divide_sections(content, complete_matches6)
        for section, content in sections_1.items():
            private_parent = parent + ' 中的 ' + section
            content = '\n' + content
            segmentation(entity, file_name, content, private_parent, id_generator, 3)
    elif len(matches7) > 0 & type != 4:
        complete_matches7 = []
        for match7 in matches7:
            complete_match7 = match7[0] + match7[1]
            complete_matches7.append(complete_match7)
        sections_1 = divide_sections(content, complete_matches7)
        for section, content in sections_1.items():
            private_parent = parent + ' 中的 ' + section
            content = '\n' + content
            segmentation(entity, file_name, content, private_parent, id_generator, 4)
    elif len(matches8) > 0 & type != 5:
        complete_matches8 = []
        for match8 in matches8:
            complete_match8 = match8[0] + match8[1]
            complete_matches8.append(complete_match8)
        sections_1 = divide_sections(content, complete_matches8)
        for section, content in sections_1.items():
            private_parent = parent + ' 中的 ' + section
            content = '\n' + content
            segmentation(entity, file_name, content, private_parent, id_generator, 5)
    elif len(matches9) > 0 & type != 6:
        complete_matches9 = []
        for match9 in matches9:
            complete_match9 = match9[0] + match9[1]
            complete_matches9.append(complete_match9)
        sections_1 = divide_sections(content, complete_matches9)
        for section, content in sections_1.items():
            private_parent = parent + ' 中的 ' + section
            content = '\n' + content
            segmentation(entity, file_name, content, private_parent, id_generator, 6)
    else:
        try:
            content = process_table(content)
        except:  # 无法处理的表格会跳过
            pass
        result = parent + ' 中提到，' + content
        result = mask_entity(entity, result)
        # 超过2048个token（约为900字）的字符串截取丢弃
        # result = result[:900]
        # 将分割好的章节存入一个文件
        current_id = id_generator.generate_id()
        with open(f'{file_name}\\{current_id}.txt', "w", encoding="utf-8") as output_file:
             output_file.write(result)



def remove_page_num(page_header, content):
    page_num_pattern = rf'\n?\d+(?:\/\d+)?\n\n{page_header}\n'
    page_num_matches = re.findall(page_num_pattern, content)
    for page_num_match in page_num_matches:
        content = content.replace(page_num_match, '')
    return content


# 先去除页眉页码，再根据目录中提取的章节标题，对string类型的文本进行切分
def process_file(file_path, page_header, section_titles, id_generator):
    # segments = []
    file_name = file_path.split('\\')[-1]
    # 若文件不存在，需要makedirs，否则不需要
    os.makedirs(file_name)
    # 获取主体名称和简称
    pattern1 = r'__(.*?)__'
    matches3 = re.findall(pattern1, file_name)
    pattern2 = r'__(\d+年)__'
    matches4 = re.findall(pattern2, file_name)
    entity = []
    for match3 in matches3:
        entity.append(match3)
    for match4 in matches4:
        entity.append(match4)
    text = open(file_path, 'r', encoding='utf-8').read()
    text = remove_page_num(page_header, text)  # 去除页眉页码
    text = text.replace("\n", "\n\n")
    sections = divide_sections(text, section_titles)  # 按目录切分文本
    for section, content in sections.items():
        parent = '在' + entity[2] + '年度报告中的 ' + section
        content = '\n' + content
        segmentation(entity, file_name, content, parent, id_generator, 0)



def prepare_data(path, size):
    batch_docs = []
    for file in os.listdir(path):
        with open(path + '/' + file, 'r', encoding='utf-8') as f:
            batch_docs.append(f.read())
            if len(batch_docs) == size:
                yield batch_docs[:]
                batch_docs.clear()
    if batch_docs:
        yield batch_docs


if __name__ == '__main__':
    id_generator = AutoIncrementIDGenerator()
    for file_name in os.listdir('..\pdf_to_txt\\test_txt'):
        if file_name.endswith('_txt.txt'):
            file_path_1 = os.path.join('..\pdf_to_txt\\test_txt', file_name)
            file_path_2 = os.path.join('..\pdf_to_txt\\test_txt', file_name.replace('_txt.txt', '.txt'))
            toc_entries = txt2toc(file_path_2)  # 获取页数和行信息
            section_titles = process_content_page(toc_entries)
            try:
                process_file(file_path_1, toc_entries[0][3], section_titles, id_generator)
            except:
                with open('segment_error.txt', "a", encoding="utf-8") as output_file:
                    output_file.write(file_name + '\n')
                pass
            id_generator.reset()

    # with open('..\\vector_store\save_error','r') as f:
    #    for line in f:
    #        file_name = line.replace('\n', '')

    # file_name = '2020-03-03__哈尔滨秋林集团股份有限公司__600891__秋林集团__2019年__年度报告_txt.txt'
    # file_path_1 = os.path.join('..\pdf_to_txt\\test_txt', file_name)
    # file_path_2 = os.path.join('..\pdf_to_txt\\test_txt', file_name.replace('_txt.txt', '.txt'))
    # toc_entries = txt2toc(file_path_2)  # 获取页数和行信息
    # section_titles = process_content_page(toc_entries)
    # process_file(file_path_1, toc_entries[0][3], section_titles, id_generator)
    # id_generator.reset()
