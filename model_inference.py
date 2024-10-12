from algorithm import SIFTMatcher

def main():
    # Initialize the SIFTMatcher
    sift_matcher = SIFTMatcher()

    # Same season example
    image_path1_same = 'images for demo/same season/T36UYA_20190825T083601_TCI.jpg'
    image_path2_same = 'images for demo/same season/T36UYA_20190904T083601_TCI.jpg'
    print("Processing same season images...")
    sift_matcher.process_images(image_path1_same, image_path2_same, max_matches=20)

    # Different season example
    image_path1_diff = 'images for demo/different seasons/T36UYA_20190726T083611_TCI.jpg'
    image_path2_diff = 'images for demo/different seasons/T36UYA_20190815T083601_TCI.jpg'
    print("Processing different season images...")
    sift_matcher.process_images(image_path1_diff, image_path2_diff, max_matches=20)

if __name__ == "__main__":
    main()
