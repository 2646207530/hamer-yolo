# 环境配置
```
pip install -e .
pip install -r requirements.txt
```

# 代码运行
```
cd hamer
python infer.py --input <RGB_dir> --output <output_dir>
```

默认储存为npy格式，提供了方法 reconstruct_and_save_obj_with_wrapper 可视化手部结果