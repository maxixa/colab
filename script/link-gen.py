links = """https://huggingface.co/Maxixa/bili/resolve/main/down.sh.zip?download=true
"""

for link in links.splitlines():
	link = link.replace("?download=true", '')
	filenames = link.split('/')
	filename = filenames[-1]
	print(f'aria2c --header "Authorization: Bearer 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" -d /data/data/com.termux/files/home/storage/external-1 -o {filename} {link}?access_token=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 2>&1 | tee aria2c.log', end='\n')




