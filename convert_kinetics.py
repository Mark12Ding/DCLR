import os
import moviepy.editor as mp
import numpy as np
from joblib import delayed
from joblib import Parallel

def downscale_video(video_file, output_file, side_size=256):

    if os.path.exists(output_file):
        return 'EXISTS'

    clip = mp.VideoFileClip(video_file)
    clip = clip.without_audio()
    short_side = np.argmin(clip.size)
    if short_side == 0:
        clip_resized = clip.resize(width=side_size)
    else:
        clip_resized = clip.resize(height=side_size)
    clip_resized.write_videofile(output_file)

    status = os.path.exists(output_file)

    return status

def convert_warpper(row, subfolder):
    video_name = row
    source_video = os.path.join(source_dir, subfolder, 'cate', video_name)
    output_video = os.path.join(dest_dir, subfolder, 'cate', video_name)
    if os.path.exists(source_video):
        # change the name of videos whose file names begin with illegal characters like "-"
        # since this will cause error in ffmpeg
#        prefix = ''
#        while video_name[0] == '-':
#            video_name = video_name[1:]
#            prefix += '-'

        try:
            downscale_video(source_video, output_video, 256)
        except:
            with open('failed_videos.txt', 'a+') as f:
                f.write(source_video+'\n')

        # shutil.copy2(source_video, os.path.join(label_dir, video_name))
        # os.symlink(source_video, os.path.join(label_dir, video_name))
        print(', '.join(row))

if __name__ == '__main__':
    source_dir = './video'
    dest_dir = './video_256'
    # source_dir = '../../Data/kinetics-400'
    # dest_dir = '../../Data/kinetics-400-tv'

    # file_list = pd.read_csv(os.path.join(source_dir, 'kinetics-400_train.csv'), header=None, sep=' ')[0]
    # status_lst = Parallel(n_jobs=16)(delayed(convert_warpper)(row, 'train') for row in file_list[1:])

    file_list = os.listdir(os.path.join(source_dir, 'train/cate'))
    status_lst = Parallel(n_jobs=16)(delayed(convert_warpper)(row, 'train') for row in file_list)
