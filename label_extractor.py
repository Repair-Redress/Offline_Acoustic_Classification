from tflite_support.metadata import MetadataDisplayer

m = MetadataDisplayer.with_model_file("yamnet.tflite")
print("Associated files:", m.get_packed_associated_file_list())

for name in m.get_packed_associated_file_list():
    buf = m.get_associated_file_buffer(name)
    with open(name, "wb") as f:
        f.write(buf)
    print("Extracted:", name)