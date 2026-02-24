# export/tflite_to_cc.py

input_path = "export/admosys_model_int8.tflite"
output_path = "export/admosys_model_int8.cc"

with open(input_path, "rb") as f:
    data = f.read()

with open(output_path, "w") as f:
    f.write("const unsigned char admosys_model[] = {\n")
    for i, b in enumerate(data):
        f.write(f"0x{b:02x}, ")
        if i % 12 == 11:
            f.write("\n")
    f.write("\n};\n")
    f.write(f"const unsigned int admosys_model_len = {len(data)};\n")

print("C array generated successfully")
print("Model size:", len(data), "bytes")
