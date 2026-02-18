import pandas as pd

def sync_metadata(merged_timestamps_path, metadata_path, source_index_column, output_path):
    """
    Synchronizes a metadata file with the final video's frame numbers.

    Args:
        merged_timestamps_path (str): Path to the 'merged_timestamps.csv' file.
        metadata_path (str): Path to the original metadata CSV file.
        source_index_column (str): The column name in the merged file that corresponds
                                   to the source video (e.g., 'index1', 'index2').
        output_path (str): Path to save the new synchronized metadata CSV.
    """
    try:
        # Load the timestamp mapping file
        merged_df = pd.read_csv(merged_timestamps_path)
        # The final frame number is simply the row index of the merged file
        merged_df['final_frame_number'] = merged_df.index

        # Load the original metadata file
        metadata_df = pd.read_csv(metadata_path)

        # Merge the metadata with the timestamp map.
        # This aligns the original 'frame_number' with the 'final_frame_number'.
        synced_df = pd.merge(
            metadata_df,
            merged_df[['final_frame_number', source_index_column]],
            left_on='frame_number',
            right_on=source_index_column,
            how='inner'  # 'inner' ensures we only keep frames present in both files
        )

        # Reorder columns to have the new frame number at the start
        # and drop the now redundant source index column
        original_cols = [col for col in metadata_df.columns if col != 'frame_number']
        final_cols = ['final_frame_number', 'frame_number'] + original_cols
        synced_df = synced_df[final_cols]
        
        # Rename original 'frame_number' to avoid confusion
        synced_df = synced_df.rename(columns={'frame_number': 'original_frame_number'})


        # Save the updated dataframe
        synced_df.to_csv(output_path, index=False)
        print(f"Successfully created synchronized metadata file at: {output_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please check your file paths.")
    except KeyError as e:
        print(f"Error: Column {e} not found. Ensure your CSVs have the correct column names ('frame_number', '{source_index_column}').")

if __name__ == '__main__':
    # --- Configuration ---
    
    # Path to the file created by your merge_timestamps function
    merged_timestamps_file = ''

    # --- Sync Wearable Glasses Metadata ---
    # Assuming 'vid1' was the wearable glasses video
    sync_metadata(
        merged_timestamps_path=merged_timestamps_file,
        metadata_path='',
        source_index_column='index1',
        output_path=''
    )

    # --- Sync GoPro Metadata ---
    # Assuming 'vid2', 'vid3', and 'vid4' were the GoPro videos
    
    # Sync GoPro 1 (corresponding to vid2)
    sync_metadata(
        merged_timestamps_path=merged_timestamps_file,
        metadata_path='',
        source_index_column='index3',
        output_path=''
    )

    # Sync GoPro 2 (corresponding to vid3)
    sync_metadata(
        merged_timestamps_path=merged_timestamps_file,
        metadata_path='',
        source_index_column='index2',
        output_path=''
    )
    
    # Sync GoPro 3 (corresponding to vid4)
    sync_metadata(
        merged_timestamps_path=merged_timestamps_file,
        metadata_path='',
        source_index_column='index4',
        output_path=''
    )