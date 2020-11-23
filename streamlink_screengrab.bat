streamlink "http://twitch.tv/forsen" best -O | ffmpeg -i pipe:0 -vf fps=fps=1/5 -update 1 screenread/img.jpg

pause