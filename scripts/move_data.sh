cd data/01_raw
echo "######################"
echo "Moving data"
echo "######################"
for ID in 1 2 3 4 5 6 7 8 9 10; do
    mv DB6_s${ID}_a/* .
    mv DB6_s${ID}_b/* .
    rm -rf DB6_s${ID}_a
    rm -rf DB6_s${ID}_b
done
cd ../..
