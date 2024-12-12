def match_string(str1: str, str2: str) -> bool:
  
    if str1 == str2:
        return True
  
    min_len = min(len(str1), len(str2))
    for i in range(min_len):
        if str1[i] != str2[i]:
            start = max(0, i - 50)
            end = min(len(str1), i + 50)
            print(f"Error at index {i}:\n'{str1[start:end]}'\n-----------------------\n'{str2[start:end]}'")
            return False
    if len(str1) != len(str2):
        start = max(0, min_len - 50)
        end = min(len(str1), min_len + 50)
        print(f"Error at index {min_len}:\n'{str1[start:end]}'\n-----------------------\n'{str2[start:end]}'")
        return False
    return False


