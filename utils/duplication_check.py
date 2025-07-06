import hashlib
import json
from datetime import datetime, date
from decimal import Decimal
from uuid import UUID
import base64
import threading

class UniqueIDEncoder(json.JSONEncoder):
    """自定义JSON编码器，用于处理各种特殊数据类型"""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode('ascii')
        if hasattr(obj, '__dict__'):
            return vars(obj)
        return super().default(obj)

def generate_data_identifier(data, *, sort_keys=True, ensure_ascii=False, indent=None):
    """
    生成数据的唯一标识符(SHA256哈希)
    
    参数:
        data: 任意可序列化的数据（字典、列表、基本类型等）
        sort_keys: 是否排序字典键（确保顺序不影响结果）
        ensure_ascii: 是否确保ASCII输出
        indent: JSON缩进（None表示紧凑格式）
        
    返回:
        str: 64字符的SHA256哈希字符串
    """
    try:
        # 序列化数据为JSON字符串
        data_str = json.dumps(
            data,
            cls=UniqueIDEncoder,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
            indent=indent
        )
        
        # 计算SHA256哈希
        sha256_hash = hashlib.sha256(data_str.encode('utf-8'))
        return sha256_hash.hexdigest()
    
    except (TypeError, ValueError) as e:
        raise ValueError(f"无法序列化数据生成标识符: {str(e)}")

# 使用示例
if __name__ == "__main__":
    # 示例1: 简单字典
    data1 = {"name": "Alice", "age": 30, "city": "New York"}
    print(f"示例1哈希: {generate_data_identifier(data1)}")
    
    # 示例2: 包含特殊类型
    data2 = {
        "event": "Meeting",
        "time": datetime(2023, 5, 15, 14, 30),
        "participants": ["Alice", "Bob"],
        "id": UUID('12345678123456781234567812345678'),
        "budget": Decimal("1250.50")
    }
    print(f"示例2哈希: {generate_data_identifier(data2)}")
    
    # 示例3: 相同数据不同顺序
    data3a = {"a": 1, "b": 2, "c": 3}
    data3b = {"c": 3, "b": 2, "a": 1}
    print(f"相同数据不同顺序是否一致: {generate_data_identifier(data3a) == generate_data_identifier(data3b)}")


def get_existing_data(output_file) -> list:
    """读取现有数据"""
    lock = threading.Lock()
    with lock:
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

def get_existng_ids(output_file) -> set:
    """获取现有数据的ID集合"""
    existing_data = get_existing_data(output_file)
    return {entry["id"] for entry in existing_data if "id" in entry}
