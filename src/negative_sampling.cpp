// -*- coding: utf-8 -*- 
// @File : negative_sampling.cpp
// @Time : 2024/01/08 17:51:37
// @Author : Amonologue 
// @Software : Visual Studio Code
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <set>


struct Rating
{
    int user;
    int item;
    float rating;
    int timestamp;
};

struct ExtendSample
{
    int user;
    int positiveItem;
    int negativeItem;
};

std::vector<Rating> loadRatings(const std::filesystem::path& filepath)
{
    std::vector<Rating> ratings;
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return ratings;
    }

    std::string line;
    while (std::getline(file, line))
    {
        int user, item, timestamp;
        float rating;
        if (sscanf(line.c_str(), "%d\t%d\t%f\t%d", &user, &item, &rating, &timestamp) == 4)
        {
            ratings.push_back({user, item, rating, timestamp});
        }
    }
    file.close();
    return ratings;
}

std::vector<ExtendSample> negativeSampling(std::vector<Rating>& ratings, const int num_negatives)
{
    int maxItem = 0;
    std::set<std::tuple<int, int> > positiveSamples;
    for (const Rating& rating : ratings)
    {
        positiveSamples.insert(std::make_tuple(rating.user, rating.item));
        maxItem = maxItem > rating.item ? maxItem : rating.item;
    }
    std::vector<ExtendSample> extendSamples;
    int cnt = 0;
    int size = ratings.size();
    for (const Rating& rating : ratings)
    {
        int user = rating.user;
        int positiveItem = rating.item;
        for (int i = 0; i < num_negatives; i ++)
        {
            int negativeItem = rand() % maxItem + 1;
            while (positiveSamples.find(std::make_tuple(user, positiveItem)) != positiveSamples.end())
            {
                negativeItem = rand() % maxItem + 1;
            }
            extendSamples.push_back({user, positiveItem, negativeItem});
        }
        std::cout << "[" << cnt ++ << "/" << size << "]" << std::endl;
    }
    return extendSamples;
}

void saveExtendSamples(const std::filesystem::path& filepath, const std::vector<ExtendSample>& extendSamples)
{
    std::ofstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "Faild to open file: " << filepath << std::endl;
        return ;
    }
    for (const ExtendSample& extendSample : extendSamples)
    {
        file << extendSample.user << "\t" << extendSample.positiveItem << "\t" << extendSample.negativeItem << std::endl;
    }
    file.close();
}

int main(int argc, char* argv[])
{
    std::string dataset = argv[1];
    std::string fold_index = argv[2];

    std::filesystem::path prefix = "data";
    std::filesystem::path suffix_load = ".train";
    std::filesystem::path suffix_save = ".extend";

    std::filesystem::path load_filepath;
    std::filesystem::path save_filepath;
    if (dataset == "amazon-music")
    {
        load_filepath = prefix / "Amazon" / "music" / ("amazon_music" + fold_index) / suffix_load;
        save_filepath = prefix / "Amazon" / "music" / ("amazon_music" + fold_index) / suffix_save;
    }
    else if (dataset == "ciao")
    {
        load_filepath = prefix / "Ciao" / ("movie-ratings" + fold_index) / suffix_load;
        save_filepath = prefix / "Ciao" / ("movie-ratings" + fold_index) / suffix_save;
    }
    else if (dataset == "douban-book")
    {
        load_filepath = prefix / "Douban" / "book" / ("douban_book" + fold_index) / suffix_load;
        save_filepath = prefix / "Douban" / "book" / ("douban_book" + fold_index) / suffix_save;
    }
    else if (dataset == "douban-movie")
    {
        load_filepath = prefix / "Douban" / "movie" / ("douban_movie" + fold_index) / suffix_load;
        save_filepath = prefix / "Douban" / "movie" / ("douban_movie" + fold_index) / suffix_save;
    }
    else if (dataset == "ml-1m")
    {
        load_filepath = prefix / "ml-1m" / ("ratings" + fold_index) / suffix_load;
        save_filepath = prefix / "ml-1m" / ("ratings" + fold_index) / suffix_save;
    }
    else if (dataset == "ml-10m")
    {
        load_filepath = prefix / "ml-10m" / ("ratings" + fold_index) / suffix_load;
        save_filepath = prefix / "ml-10m" / ("ratings" + fold_index) / suffix_save;
    }
    std::cout << "Load path: " << load_filepath << std::endl;
    std::cout << "Save path: " << save_filepath << std::endl;
    std::vector<Rating> ratings = loadRatings(load_filepath);
    if (ratings.empty())
    {
        std::cerr << "No ratings data loaded." << std::endl;
        return 1;
    }
    std::vector<ExtendSample> extendSamples = negativeSampling(ratings, 4);
    if (extendSamples.empty())
    {
        std::cerr << "Generate ExtendSample error." << std::endl;
        return 1;
    }
    saveExtendSamples(save_filepath, extendSamples);
    return 0;
}