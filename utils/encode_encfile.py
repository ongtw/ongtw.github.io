import base64

for i in range(1,79):
    i_str = str(i).zfill(2)
    infile = f"model.7z.0{i_str}"
    outfile = f"e{i_str}.html"
    print(i, infile, outfile)

    with open(infile, "rb") as f:
        enc = base64.b64encode(f.read())

    with open(outfile, "w") as f:
        f.write("<html><body>\n")
        f.write(enc.decode("utf-8"))
        f.write("\n</body></html>")

