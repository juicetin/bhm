path=dm24_argmax_images
counter=1
gif_count=0
cur_files=""
for img in $path/*.pdf
do
    counter=$((counter+1))
    cur_files+=" $(ls $img)"
    if [ $counter -gt 200 ]; then
        convert -verbose $cur_files "dm24_argmaxes_$gif_count.gif"
        cur_files=""
        gif_count=$((gif_count+1))
        counter=0
    fi
done

# echo $str
echo $cmd
