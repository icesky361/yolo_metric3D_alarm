#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集划分工具（K折交叉验证版）

主要功能：
1. 支持K折交叉验证的数据集划分（默认10折）
2. 基于组合特征（监拍用途>安装介质>设备型号>生产厂家）进行分层抽样
3. 每折按7:2:1比例划分train/val/test子集
4. 处理空值数据并生成详细划分日志
5. 结果包含多列数据集分类（每折对应一列）

核心模块关系：
- main(): 程序入口，控制交叉验证流程
- split_dataset_improved(): 实现K折交叉验证核心逻辑
- create_extended_composite_feature(): 创建组合特征用于分层抽样
- StratifiedKFold: 实现分层K折交叉验证划分
- find_excel_files()/select_excel_file(): 辅助Excel文件选择
"""
import os
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import random
import logging
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

def find_excel_files():
    """查找当前目录下的Excel文件(.xls或.xlsx)
    
    返回:
        list: 找到的Excel文件名列表
    """
    excel_files = []
    for file in os.listdir(os.path.dirname(__file__)):
        if file.lower().endswith(('.xls', '.xlsx')):
            excel_files.append(file)
    return excel_files

def select_excel_file():
    """让用户从Excel文件列表中选择数据集文件
    
    返回:
        str: 选中的Excel文件名，未找到则返回None
    """
    excel_files = find_excel_files()
    if not excel_files:
        print("当前目录下没有找到Excel文件(.xls或.xlsx)")
        return None
    
    print("请选择数据集文件:")
    for i, file in enumerate(excel_files, 1):
        print(f"{i}. {file}")
    
    while True:
        try:
            choice = int(input("请输入序号: "))
            if 1 <= choice <= len(excel_files):
                return excel_files[choice - 1]
            print("序号无效，请重新输入")
        except ValueError:
            print("请输入有效的数字序号")

def stratified_split(df, group_column, split_ratio=(0.7, 0.2, 0.1)):
    """按照指定列进行分层抽样，尽量在每个类别内部按7:2:1划分"""
    if sum(split_ratio) != 1:
        raise ValueError("Split ratios must sum to 1")

    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()

    # 使用 groupby 处理每个组合特征类别
    for group_name, group_data in df.groupby(group_column):
        group_data = group_data.sample(frac=1, random_state=42).reset_index(drop=True)
        n_total = len(group_data)

        # 计算理想的分割点
        n_train_ideal = n_total * split_ratio[0]
        n_val_ideal = n_total * split_ratio[1]
        # n_test_ideal = n_total * split_ratio[2] # 测试集是剩余的

        # 实际分配数量，优先保证整数个，然后处理余数
        n_train = int(round(n_train_ideal))
        n_val = int(round(n_val_ideal))
        n_test = n_total - n_train - n_val

        # 处理极端情况，例如n_test为负数或非常小的情况
        if n_test < 0:
            # 如果测试集数量为负，说明训练集和验证集分配过多
            # 优先从验证集减少，再从训练集减少
            if n_val + n_test < 0: # 如果验证集减去后还不够
                n_train += (n_val + n_test) # 将不足的部分从训练集减去（n_val+n_test是负数）
                n_val = 0
            else:
                n_val += n_test # 将测试集的负数部分加到验证集（实际上是减少验证集）
            n_test = 0
        
        # 确保训练集和验证集至少有1个（如果总数允许）
        if n_total > 0:
            if n_train == 0 and n_total > 0 : n_train = 1
            if n_val == 0 and n_total > n_train : n_val = 1
        
        # 重新计算测试集数量，并确保非负
        n_test = n_total - n_train - n_val
        if n_test < 0: # 再次检查，如果因为之前的调整导致测试集为负
            if n_val > 0 : # 从验证集挪一个给测试集
                n_val -=1
                n_test +=1
            elif n_train > 1: # 如果验证集没了，从训练集挪（确保训练集至少留一个）
                n_train -=1
                n_test +=1
            # 如果总共就1条，那它会在train里，test和val都是0，这是期望的

        # 最终分配，确保索引不越界
        current_pos = 0
        if n_train > 0:
            train_df = pd.concat([train_df, group_data.iloc[current_pos:current_pos + n_train]])
            current_pos += n_train
        if n_val > 0 and current_pos < n_total:
            val_df = pd.concat([val_df, group_data.iloc[current_pos:min(current_pos + n_val, n_total)]])
            current_pos += n_val
        if n_test > 0 and current_pos < n_total:
            test_df = pd.concat([test_df, group_data.iloc[current_pos:n_total]])
            
    return train_df, val_df, test_df

def create_composite_feature(row, features_priority):
    """创建组合特征，按照重要性排序"""
    values = []
    for feature in features_priority:
        if feature in row and not pd.isna(row[feature]):
            # 安装介质的空值按照路灯杆处理
            if feature == '安装介质' and (pd.isna(row[feature]) or str(row[feature]).strip() == ''):
                values.append('路灯杆')
            else:
                values.append(str(row[feature]).strip())
        else:
            # 安装介质的空值按照路灯杆处理
            if feature == '安装介质':
                values.append('路灯杆')
            else:
                values.append('未知')
    return "_".join(values)

def create_extended_composite_feature(df):
    """创建扩展组合特征用于分层抽样
    
    组合多个关键特征生成唯一标识，处理空值并添加随机扰动
    
    参数:
        df (pd.DataFrame): 包含设备数据的DataFrame
    
    返回:
        pd.DataFrame: 添加了'composite_feature'列的DataFrame
    """
    # 处理空值，用'Unknown'替换
    column_mapping = {
        '监拍用途': ['监拍用途', '通道监拍', '监拍类型'], # 优先使用“监拍用途”
        '安装介质': ['安装介质'],
        '设备型号': ['装置型号', '设备型号', '型号'],
        '生产厂家': ['生产厂家']
    }
    
    # 严格按照用户要求的顺序
    ordered_features_to_check = ['监拍用途', '安装介质', '设备型号', '生产厂家']
    
    features_for_composite = [] # 存储实际用于创建组合特征的列名
    actual_columns_used_log = [] # 用于日志记录

    for feature_key in ordered_features_to_check:
        found_actual_column = False
        for actual_col_name in column_mapping.get(feature_key, []):
            if actual_col_name in df.columns:
                features_for_composite.append(actual_col_name)
                actual_columns_used_log.append(actual_col_name)
                found_actual_column = True
                break
        if not found_actual_column:
            # 如果是安装介质且未找到，仍然要包含它，create_composite_feature会处理空值
            if feature_key == '安装介质':
                features_for_composite.append('安装介质') # 即使列不存在，也加入，以便create_composite_feature统一处理
                actual_columns_used_log.append(f'{feature_key}(缺失或按默认处理)')
            else:
                # 对于其他未找到的字段，也添加一个占位符，以便create_composite_feature统一处理为'未知'
                features_for_composite.append(feature_key) # 使用原始的key作为占位符
                actual_columns_used_log.append(f'{feature_key}(缺失)')
                print(f"警告：字段 '{feature_key}' (或其替代字段) 未在Excel中找到。组合时将视为'未知'。")

    print(f"组合特征将基于以下字段（按重要性排序生成）: {actual_columns_used_log}")

    if not features_for_composite: # 如果一个字段都没匹配上（理论上安装介质会保证至少有一个）
        print("警告：未能找到任何有效字段来创建组合特征。将使用默认占位符。")
        df['组合特征'] = '无有效特征组合'
        return []

    df['组合特征'] = df.apply(lambda row: create_composite_feature(row, features_for_composite), axis=1)
    return features_for_composite # 返回实际用于组合的列名列表

def split_dataset_improved(excel_path, n_splits=10):
    """改进的数据集划分方法，使用组合特征和分步骤分配，支持K折交叉验证"""
    try:
        # 创建日志目录和日志文件
        log_dir = os.path.join(os.path.dirname(__file__), 'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = os.path.join(log_dir, f"数据集划分_{datetime.now().strftime('%Y%m%d%H%M')}.log")
        logging.basicConfig(filename=log_file, level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        
        # 读取Excel文件
        df_original = pd.read_excel(excel_path)
        logging.info(f"读取Excel文件: {excel_path}, 总行数: {len(df_original)}")

        # 筛选在运设备和库存设备
        if '设备状态' in df_original.columns:
            df_in_operation = df_original[df_original['设备状态'] == '在运'].copy()
            df_inventory = df_original[df_original['设备状态'] != '在运'].copy()
            logging.info(f"在运设备数量: {len(df_in_operation)}, 库存设备数量: {len(df_inventory)}")
        else:
            df_in_operation = df_original.copy()
            df_inventory = pd.DataFrame()
            logging.warning("Excel文件中没有'设备状态'列，将处理所有数据")

        # 定义关键字段
        key_columns = ['大屏编号', '设备编码', '监拍用途', '安装介质', '装置型号', '生产厂家']
        
        # 确保Excel中实际存在的关键字段才用于筛选
        actual_key_columns = [col for col in key_columns if col in df_in_operation.columns]
        if len(actual_key_columns) < len(key_columns):
            missing_cols = set(key_columns) - set(actual_key_columns)
            logging.warning(f"以下关键字段在Excel中不存在: {missing_cols}")

        # 步骤0: 筛选出在指定关键字段中存在空值的行
        if actual_key_columns:
            condition = df_in_operation[actual_key_columns].isnull().any(axis=1)
            df_empty_values = df_in_operation[condition].copy()
            df = df_in_operation[~condition].copy()
            
            dropped_row_count = len(df_empty_values)
            if dropped_row_count > 0:
                logging.info(f"已将{dropped_row_count}行关键字段空值记录移至'未划分数据集的空值'页签")
        else:
            logging.warning("没有有效的关键字段用于空值筛选，所有数据将用于划分")
            df = df_in_operation.copy()
            df_empty_values = pd.DataFrame()

        # 创建结果DataFrame，用于存储所有折叠的分类结果
        result_df = df.copy()
        
        # 步骤1: 创建扩展的组合特征
        features_used = create_extended_composite_feature(result_df)
        
        if features_used and '组合特征' in result_df.columns:
            # 使用StratifiedKFold进行K折交叉验证
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            # 确保组合特征列是字符串类型，以便StratifiedKFold正确处理
            result_df['组合特征'] = result_df['组合特征'].astype(str)

            for fold, (train_val_index, test_index) in enumerate(skf.split(result_df, result_df['组合特征'])):
                fold_col_name = f'数据集分类{fold + 1}'
                result_df[fold_col_name] = None # 初始化当前折叠的分类列
                
                # 将当前折叠的测试集标记为'test'
                result_df.loc[result_df.index[test_index], fold_col_name] = 'test'
                
                # 从剩余的训练验证集中划分训练集和验证集
                train_val_df = result_df.iloc[train_val_index].copy()
                
                # 再次创建组合特征，以防在子集中有变化（尽管这里应该保持一致）
                # create_extended_composite_feature(train_val_df) # 这一步不需要，因为组合特征已经创建

                # 对训练验证集进行分层抽样，划分为train和val
                # 这里需要确保train_val_df有足够的样本进行划分
                if not train_val_df.empty:
                    # 重新计算train和val的比例，使其在train_val_df中是7:2
                    # 原始比例是7:2:1，现在测试集已经分出1，所以训练验证集总共是9
                    # 那么在训练验证集中，训练集占7/9，验证集占2/9
                    train_val_split_ratio = (0.7 / (0.7 + 0.2), 0.2 / (0.7 + 0.2))
                    
                    # 确保组合特征列是字符串类型
                    train_val_df['组合特征'] = train_val_df['组合特征'].astype(str)

                    # 使用groupby进行分层抽样
                    train_indices_fold = []
                    val_indices_fold = []

                    for feature, group in train_val_df.groupby('组合特征'):
                        n_samples_group = len(group)
                        indices_group = list(group.index)
                        random.shuffle(indices_group)

                        n_train_group = int(round(n_samples_group * train_val_split_ratio[0]))
                        n_val_group = n_samples_group - n_train_group

                        # 确保至少有一个样本分配给训练集，如果总数允许
                        if n_samples_group > 0 and n_train_group == 0: n_train_group = 1
                        # 确保至少有一个样本分配给验证集，如果总数允许且训练集已分配
                        if n_samples_group > n_train_group and n_val_group == 0: n_val_group = 1
                        
                        # 重新调整以确保总和正确
                        if n_train_group + n_val_group != n_samples_group:
                            n_train_group = n_samples_group - n_val_group
                            if n_train_group < 0: # 极端情况，如果验证集分配过多
                                n_val_group += n_train_group # 减少验证集
                                n_train_group = 0
                            if n_train_group == 0 and n_samples_group > 0: n_train_group = 1 # 确保训练集至少一个
                            n_val_group = n_samples_group - n_train_group # 重新计算验证集

                        train_indices_fold.extend(indices_group[:n_train_group])
                        val_indices_fold.extend(indices_group[n_train_group:n_train_group + n_val_group])
                    
                    # 标记当前折叠的训练集和验证集
                    result_df.loc[train_indices_fold, fold_col_name] = 'train'
                    result_df.loc[val_indices_fold, fold_col_name] = 'val'
                
                # 记录当前折叠的划分概览
                total_fold = len(result_df)
                train_count_fold = len(result_df[result_df[fold_col_name] == 'train'])
                val_count_fold = len(result_df[result_df[fold_col_name] == 'val'])
                test_count_fold = len(result_df[result_df[fold_col_name] == 'test'])
                
                overview_fold = f"""折叠 {fold + 1} 数据集划分概览: (基于在运设备总数据量: {total_fold})
                训练集(train): {train_count_fold} ({train_count_fold/total_fold*100:.2f}%)
                验证集(val): {val_count_fold} ({val_count_fold/total_fold*100:.2f}%)
                测试集(test): {test_count_fold} ({test_count_fold/total_fold*100:.2f}%)"""
                logging.info(overview_fold)
                print("\n" + overview_fold)

        # 保存结果到新的Excel文件
        output_filename = os.path.splitext(excel_path)[0] + "_交叉验证划分结果.xlsx"
        with pd.ExcelWriter(output_filename) as writer:
            result_df.to_excel(writer, sheet_name='划分结果', index=False)
            
            if not df_empty_values.empty:
                df_empty_values.to_excel(writer, sheet_name='未划分数据集的空值', index=False)
                
            if not df_inventory.empty:
                df_inventory.to_excel(writer, sheet_name='库存设备', index=False)
                logging.info(f"已添加'库存设备'页签，包含{len(df_inventory)}条记录")
                
        logging.info(f"结果已保存到: {output_filename}")
        return result_df, output_filename
        
    except Exception as e:
        logging.error(f"处理Excel文件时出错: {str(e)}")
        return None, None

def main():
    """程序主入口
    
    执行流程:
    1. 查找并选择Excel数据集文件
    2. 设置交叉验证折数
    3. 调用改进版划分函数执行K折分层抽样
    4. 将含多折标记的结果保存为新Excel文件
    """
    print("=== K折交叉验证数据集划分工具 ===")
    print("本工具将对在运设备数据进行K折交叉验证划分，默认为5折。")
    print("每折将尽量在每个组合特征内部，按7:2:1的比例将数据划分为train、val和test三个子集。")
    print("组合特征基于以下字段（按重要性排序）：监拍用途 > 安装介质 > 设备型号 > 生产厂家。")
    print("'安装介质'的空值（或列不存在时）将按'路灯杆'处理，其他缺失字段按'未知'处理。")
    print("划分结果将包含'组合特征'列和多列'数据集分类X'（X为折叠序号）。")
    print("不再单独生成仅含组合特征的Excel文件。")
    print("-" * 30)
    
    # 选择Excel文件
    excel_path = select_excel_file()
    if not excel_path:
        return
    
    # 进行数据集划分 (默认为10折)
    result_df_final, output_path = split_dataset_improved(excel_path, n_splits=10)
    
    if output_path and result_df_final is not None:
        print("\n处理完成！")
    elif result_df_final is None:
        print("\n处理失败，未生成结果文件。")

if __name__ == '__main__':
    main()