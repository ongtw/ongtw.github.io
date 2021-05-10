import base64

infile = "model.txt"
outfile = "model_out.tgz"

data = open(infile, "r").read()
dec = base64.b64decode(data)

output_file = open(outfile, "wb")
output_file.write(dec)
output_file.close()

