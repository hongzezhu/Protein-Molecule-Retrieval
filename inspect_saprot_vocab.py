from transformers import AutoTokenizer
import os

# 1. 加载 Tokenizer

model_path = "westlake-repl/SaProt_650M_AF2" 
print(f">>> Loading tokenizer from {model_path}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit()

# 2. 定义搜索空间
amino_acids = "ACDEFGHIKLMNPQRSTVWY" # 20种标准氨基酸
foldseek_chars = "abcdefghijklmnopqrstuvwxyz" # 可能的结构字符

print("\n>>> 正在扫描词表，寻找合法的 [AA+Struct] 组合...")
print("="*60)

# 3. 寻找万能补丁
universal_struct = None

# 我们先打印几个来看看 Token 长什么样
print("Previewing some valid tokens for 'M' (Methionine):")
example_found = False
for char in foldseek_chars:
    # 测试几种常见格式
    candidates = [f"M{char}", f"M#{char}", f"{char}M"]
    for c in candidates:
        ids = tokenizer.encode(c, add_special_tokens=False)
        if len(ids) == 1 and 3 not in ids: # id 3 usually UNK
            print(f"  ✅ Valid Token Found: '{c}' -> ID {ids[0]}")
            example_found = True
if not example_found:
    print("  ❌ 警告：未找到任何 'M' 的组合 Token，可能需要检查 vocab.txt 文件内容")

print("-" * 60)

# 4. 寻找一个对所有 20 种氨基酸都有效的结构字符
for s in foldseek_chars:
    is_universal = True
    # 假设格式是 "Mc" (大写AA + 小写Struct)，这是最常见的 Foldseek-PLM 格式
    # 我们也会测试 "M#c"
    
    # 自动探测格式
    format_template = None
    
    # 先探测格式
    test_aa = "M"
    if tokenizer.convert_tokens_to_ids(f"{test_aa}{s}") != tokenizer.unk_token_id:
        format_template = "{aa}{s}" # 格式如 Mc
    elif tokenizer.convert_tokens_to_ids(f"{test_aa}#{s}") != tokenizer.unk_token_id:
        format_template = "{aa}#{s}" # 格式如 M#c
    
    if not format_template:
        continue # 这个结构字符 s 连 M 都匹配不上，跳过

    # 验证是否覆盖所有氨基酸
    for aa in amino_acids:
        token = format_template.format(aa=aa, s=s)
        idx = tokenizer.convert_tokens_to_ids(token)
        if idx == tokenizer.unk_token_id:
            is_universal = False
            break
    
    if is_universal:
        print(f"\n🎉 找到了万能结构后缀: '{s}'")
        print(f"🎉 确定的 Token 格式: '{format_template.format(aa='M', s=s)}'")
        print(f"   (这意味着你可以用 '{format_template.format(aa='aa', s=s)}' 来补全所有序列)")
        universal_struct = (s, format_template)
        break

if not universal_struct:
    print("\n❌ 未找到单一的万能结构字符。我们需要建立一个映射表。")
    print("正在生成安全映射表...")
    safe_map = {}
    for aa in amino_acids:
        for s in foldseek_chars:
            # 优先测试无井号格式 "Mc"
            token = f"{aa}{s}"
            if tokenizer.convert_tokens_to_ids(token) != tokenizer.unk_token_id:
                safe_map[aa] = token
                break
            # 测试带井号 "M#c"
            token_hash = f"{aa}#{s}"
            if tokenizer.convert_tokens_to_ids(token_hash) != tokenizer.unk_token_id:
                safe_map[aa] = token_hash
                break
        
        if aa not in safe_map:
            print(f"  ⚠️ 警告: 氨基酸 {aa} 找不到任何合法结构 Token！")
    
    print("\n✅ 生成了安全映射 map (部分展示):")
    print(list(safe_map.items())[:5])
