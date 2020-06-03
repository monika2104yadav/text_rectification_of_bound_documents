clear;
ori_image   = imread('partial_verify_text.jpg');
image       = rgb2gray(ori_image);
%image       = ori_image;
pr_image    = image;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%median filter%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%filter%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% med_image = pr_image;
% size_image = size(med_image);
% size_i_image = size_image(1);
% size_j_image = size_image(2);
% for i = 2:size_i_image-1
%     for j = 2:size_j_image-1
%         block = reshape(med_image(i-1:i+1,j-1:j+1),1,9);
%         med   = median(block);
%         med_image(i,j) = med;
%     end
% end
% figure('Name','Median Filtered Image');imshow(med_image);
% pr_image = med_image;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%median filter%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%filter%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

hist_image  = adapthisteq(pr_image);
pr_image    = hist_image;
figure('Name','Equalized Image');imshow(hist_image);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% obtaining curve: open-close followed by Niblacks. - issue many white
%% curved lines however with enough noise obtained.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
line_struct = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1];
% size_line   = size(line_struct);
% line_struct   = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;
%                1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;
%                1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;
%                1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;
%                1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;];
% size_sq     = size(sq_struct);
% op_image = imopen(pr_image,sq_struct);
% cl_op_image = imclose(pr_image,line_struct);
% imshow(cl_op_image);
% pr_image = cl_op_image;
% %%%%%%%%%%%%%%%%Niblack's method%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% size_line   = size(line_struct);
% hs_struct_x = floor(size_line(1)/2);
% hs_struct_y = floor(size_line(2)/2);
% for i = hs_struct_x+1:size_image(1)-hs_struct_x
%    for j = hs_struct_y+1:size_image(2)-hs_struct_ymage
%        win_image = pr_image(i - hs_struct_x : i + hs_struct_x, j - hs_struct_y : j + hs_struct_y);
%        win_image = reshape(win_image, 1, size_line(1) * size_line(2));
%        mean_win  = mean(win_image);
%        std_win   = std(double(win_image));
%        thresh_win= mean_win - 0.2*std_win;
%        if pr_image(i,j) > thresh_win
%           ni_image(i,j) = 255;
%        else
%           ni_image(i,j) = 0;
%        end
%    end
% end
% figure('Name','niblack thresholded open close image');imshow(ni_image);  
% pr_image = ni_image;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% erosion of the image for large pages
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

size_image = size(pr_image);
size_x_image = size_image(1);
%TODO : doesn't work for half images
%if size_x_image > 1000
% line_struct=[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;
%              1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;
%              1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;
%              1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;
%              1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;
%              1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;
%              1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;
%              1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;
%              1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;];
%end
% if size_x_image > 500
%     line_struct = [1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1;
%                    1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1;];
% else
%     line_struct = [1 1 1; 1 1 1; 1 1 1;];
% end
er_image = imerode(pr_image,line_struct);
imshow(er_image);
pr_image = er_image;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% forming binary image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
size_image = size(pr_image);
size_x_image = size_image(1);
size_y_image = size_image(2);
for j = 1:size_y_image
    for i = 2:size_x_image
        if er_image(i,j) < 100
            bin_image(i,j) = 0;
        else
            bin_image(i,j) = 255;
        end
    end
end
figure('Name','Binary Image');imtool(bin_image);
pr_image = bin_image;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% obtaining the wave : first difference from pixel count
%% traverse columns till the black pixel is obtained. and later half
%% discusses to place the curves based on horizontal values. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% size_pr_image = size(pr_image);
% size_x_image = size_pr_image(1);
% size_y_image = size_pr_image(2);
% count = 0;
% for i = 1:size_x_image
%     for j = 1:size_y_image
%         if pr_image(i,j)== 0
%             j_for_first_non_zero_i_s(count + 1) = j;
%             first_non_zero_i_s(count + 1) = i;
%             if count > 0 
%                diff_first_non_zero_i_s = (first_non_zero_i_s(count + 1) - first_non_zero_i_s(count));
%                if diff_first_non_zero_i_s < 10
%                    diff_j_for_first_non_zero_i_s = abs(j_for_first_non_zero_i_s(count+1) - j_for_first_non_zero_i_s(count));
%                    if diff_j_for_first_non_zero_i_s > size_y_image/4
%                       A = j_for_first_non_zero_i_s(count+1);
%                       B = j_for_first_non_zero_i_s(count);
%                       min_j_val = min(A,B);
%                       if min_j_val == j_for_first_non_zero_i_s(count+1)
%                          %so the prev. value may not help in app. curve, remove it.
%                          j_for_first_non_zero_i_s(count) = j_for_first_non_zero_i_s(count + 1);
%                          first_non_zero_i_s(count) = first_non_zero_i_s(count+1);
%                       end
%                       if count > 0
%                          count = count - 1; %remove the current value as it is not useful.5
%                       end
%                    end
%                    if diff_j_for_first_non_zero_i_s < 2
%                        if count > 0
%                           count = count - 1;
%                        end
%                    end
%                end
%             end                
%             count = count + 1;
%             break;
%         end
%     end
% end
% for j = 1:size_y_image
%     for i = 1:size_x_image
%         if pr_image(i,j) > 0
%             continue;
%         end
%         step = 1;
%         size_j_for   = size(j_for_first_non_zero_i_s);
%         size_j_j_for = size_j_for(2);
%         matching_first_val_found = 0;
%         for first_j = 1:size_j_j_for
%               first_j_val = j_for_first_non_zero_i_s(1,first_j);
%               first_i_val = first_non_zero_i_s(1,first_j);
%               if ((j > first_j_val || j == first_j_val) && (i < first_i_val || i == first_i_val))
%                  matching_first_val_found = 1;
%                  break;
%               end
%         end
%         if matching_first_val_found == 0
%             continue;
%         end
%         value_stored         = first_non_zero_i_s(1,first_j) - i;
%         if j > 1
%            size_average_wave = size(average_wave);
%            size_j_average_wave = size_average_wave(2);
%            if j > size_j_average_wave
%               average_wave(1,j) = (value_stored)
%            else
%               average_wave(1,j) = (average_wave(1,j) + value_stored)/2
%            end
%         else
%            average_wave(1,j) = value_stored
%         end
%         wave_image(i,j) = 1;
%     end
% end   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% obtaining the wave : pixel count
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% size_image = size(bin_image);
% size_x_image = size_image(1);
% size_y_image = size_image(2);
% for j = 1:size_y_image
%     black_pix_count(j) = 0;
%     black_pix_ival(j) = 0;
%     for i = 2:size_x_image
%         if bin_image(i,j) == 0
%            black_pix_count(j) = black_pix_count(j) + 1;
%            black_pix_ival(j)  = black_pix_ival(j) + i;
%     end
%     wave_plot(j) = black_pix_ival(j)/black_pix_count(j);
%     end
% end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% obtaining the curve: NL Means : trials of structuring elements
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this one yielded sparsed lines with a hard deduction of curve 25x30%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% obtaining the wave : component selection
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% size_image   = size(pr_image);
% size_x_image = size_image(1);
% size_y_image = size_image(2);
% 
% wave_plot_number = 1;
% processed_i = 1;
% for i = 2:size_x_image-1
%     if processed_i > i
%         continue;
%     end 
%     edge_exist = 0;
%     processed_i  = i;
%     processed_j  = 1;
%     continued_j_search = 0;
%     inc_wave_plot_number = 0;
%     for j = 1:size_y_image
%         if processed_j > j
%             continue;
%         end
%         processed_j = j;
%         %%%%%%%%%%check for value of point where base line starts%%%%%%%%%%
%         if continued_j_search == 0
%             edge_exist = abs(pr_image(i,j) - pr_image(i-1,j));
%             if edge_exist > 0
%                 continuous_region = abs(pr_image(i+1,j)-pr_image(i,j));
%                 found_once = 0;
%                 for processed_i = processed_i:size_x_image-1
%                     if continuous_region > 0
%                         if found_once == 0
%                             found_once = 1;
%                         else
%                             break;
%                         end
%                     end
%                     if processed_i > size_x_image || processed_i == size_x_image
%                         break;
%                     end
%                     continuous_region = abs(pr_image(processed_i + 1, j) - pr_image(processed_i, j));
%                 end
%                 processed_i = processed_i - 1; %decrement to get the black pixel
%             end
%         end
%         %%obtaining a baseline curve over evaluated starting edge point%%%
%         if edge_exist > 0 || continued_j_search == 1
%             for black_pix_i_search = 0:-4:-1
%                 pix_i_check = processed_i + black_pix_i_search;
%                 if pix_i_check < 1
%                    continue;
%                 end
%                 for black_pix_j_search = 0:4
%                     pix_j_check = processed_j + black_pix_j_search;
%                     if pix_j_check > size_y_image
%                         break;
%                     end
%                     if pr_image(pix_i_check, pix_j_check) == 0
%                         wave_plot(1,pix_j_check,wave_plot_number) = pix_i_check;
%                         continued_j_search = 1;
%                         inc_wave_plot_number = 1;
%                         break;
%                     end
%                 end
%             end
%             processed_i = pix_i_check;
%             processed_j = pix_j_check;
%         end
%     end
%     size_wave    = size(wave_plot);
%     size_y_wave  = size_wave(2);
%     if inc_wave_plot_number == 1
%         zero_count = 0;
%         for j = 1:size_y_wave
%             if wave_plot(1,j,wave_plot_number) > 0
%                 ;
%             else
%                 zero_count = zero_count + 1;
%             end
%         end
%         if zero_count > 10
%             continue;
%         end            
%         
%         max_wave_val = max(wave_plot(1,:,wave_plot_number));
%         for j = 1:size_y_wave
%             norm_wave_plot(1,j,wave_plot_number) = wave_plot(1,j,wave_plot_number)/max_wave_val;
%         end
%         y_axis = 1:1:size_y_image;
%         y_wave = 1:1:size_y_wave;
%   
%         wave_plot_full = interp1(y_wave,norm_wave_plot(1,:,wave_plot_number),y_axis,'nearest');
%         for j = 2:size_y_wave 
%             if wave_plot_number == 1
%                average_wave(1,j) = norm_wave_plot(1,j)
%             else
%                size_average_wave   = size(average_wave);
%                size_y_average_wave = size_average_wave(2);
%                if j > size_y_average_wave
%                    average(1,j)  = wave_plot_full(1,j)
%                else
%                    average_wave(1,j) = (wave_plot_full(1,j) + average_wave(1,j))/2
%                end
%             end
%         end
%         wave_plot_number = wave_plot_number + 1;
%     end
% end
% for j = 1:size_y_image
%     average_wave(1,j) = 1/average_wave(1,j);
% end        
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% obtaining the curve : taking a simple summation - works well as more no.
% %% pixels are bulged and have higher values than the case where lower is th
% %% bulge
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% size_image = size(pr_image);
% size_x_image = size_image(1);
% size_y_image = size_image(2);
% for j = 1:size_y_image
%     count(j) = 0;
%     i_values(j) = 0;
%     for i = 2:size_x_image
%         if pr_image(i,j) == 0
%            count(j)    = count(j) + 1;
%            i_values(j) = i_values(j) + i;
%         end
%     end
%     wave_plot(j) = i_values(j)/count(j);
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%removal of border lines
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%for j = percent_removal_from_edges:size_y_image-percent_removal_from_edges
size_image = size(pr_image);
size_x_image = size_image(1);
size_y_image = size_image(2);
for j = 1:size_y_image
    vert_space_count(1,j) = 0;
end
max_space_count = 0;
for j = 1:size_y_image
    for i = 1:size_x_image
        if pr_image(i,j) == 255
           vert_space_count(1,j) = vert_space_count(1,j) + 1;
        end
    end
    if vert_space_count(1,j) > max_space_count
        max_space_count = vert_space_count(1,j);
        pt_of_max_space = j;
    end
end

percent_border = size_y_image/10;
if pt_of_max_space < percent_border
    new_pr_image = pr_image(:,percent_border:size_y_image);
    right_border_found = 0;
    pr_image = new_pr_image;
end
if pt_of_max_space > size_y_image - percent_border
    new_pr_image = pr_image(:,1:size_y_image - percent_border);
    right_border_found = 1;
    pr_image = new_pr_image;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% obtaining waveform: choosing middle point of pixel values
%% to avoid obtaining the waveform based on upper variations in letters
%% the NL Means was tried however it didn't work better as the image also 
%% gets bit more sparsed.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

size_image = size(pr_image);
size_x_image = size_image(1);
size_y_image = size_image(2);
for j = 1:size_y_image
    top_pix_of_letter = 0;
    letter_size   = 0;
    for i = 2:size_x_image
        diff_im_intensity = pr_image(i,j) - pr_image(i-1,j);
        if diff_im_intensity < 0
            top_pix_of_letter = i;
        end
        if top_pix_of_letter > 0
            if diff_im_intensity == 0
                letter_size = letter_size + 1;
            else
                wave_image(i,j)   = top_pix_of_letter + (letter_size/2);
                letter_size       = 0;
                top_pix_of_letter = 0;
            end
        end
    end
end

figure('Name','Wave Image');imshow(wave_image);                                
%%%%%%%%%%%%%%%%%%% waveform normalization %%%%%%%%%%%%%%%%%%%%%%%%%%%%
size_wave = size(wave_image);
size_y_wave_image = size_wave(2);
size_x_wave_image = size_wave(1);

%initialization.
for j = 1:size_y_wave_image
    sum_average_wave(1,j) = 0;
end
%addition of pixel values 
for j = 1:size_y_wave_image
    for i = 1:size_x_wave_image 
        if wave_image(i,j) > 0
            if i > size_x_wave_image/2
               sum_average_wave(1,j) = sum_average_wave(1,j) + size_x_wave_image - i;
            else
               sum_average_wave(1,j) = sum_average_wave(1,j) + i;
            end
        end
    end
end

max_average_wave = max(sum_average_wave);
average_wave = sum_average_wave./max_average_wave;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% waveform plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
size_wave = size(average_wave);
size_y_wave = size_wave(2);
y_axis = 1:1:size_y_wave;
figure('Name','Plot of averaged wave');plot(y_axis,average_wave);

poly_wave     = polyfit(y_axis,average_wave,2); 
poly_val_wave = polyval(poly_wave,y_axis);
figure('Name','Plot of smoothed averaged wave');plot(y_axis,poly_val_wave,'b-') ;
min_poly_val_wave = min(poly_val_wave);
for j = 1:size_y_wave
    poly_val_wave(1,j)     = poly_val_wave(1,j)-min_poly_val_wave;
end
figure('Name','Plot of smoothed normalized to zero averaged wave');plot(y_axis,poly_val_wave,'b-');
%yielding interrupts. - to avoid interruption, try simply 
%interpolating the given wave. For right border the extra expansion
%can't be helped and for left border it is better that things on 
%right are expanded.
size_image = size(image);
size_y_image = size_image(2);
actual_y_axis = 1:1:size_y_image;

% avoided it resulted in abrupts.
% if right_border_found == 1
%     for j = size_y_image - percent_border : size_y_image
%         poly_val_wave(1,j) = poly_val_wave(size_y_wave);
%     end
% else   
%     for j = size_y_image:-1:2
%         if j > size_y_wave          
%            poly_val_wave(1, j) = poly_val_wave(size_y_wave-(size_y_image - j));
%         else
%           if j > percent_border
%               poly_val_wave(1,j) = poly_val_wave(1,j-1);
%           else
%              poly_val_wave(1,j) = 0;
%           end
%         end
%    end
%end
poly_val_wave = interp1(y_axis,poly_val_wave,actual_y_axis,'spline');
figure('Name','Plot of final smoothed averaged wave');plot(actual_y_axis,poly_val_wave);

% 
% size_poly_val_wave   = size(poly_val_wave);
% size_y_poly_val_wave = size_poly_val_wave(2);
% size_image           = size(image);
% size_y_image         = size_image(2);
% if size_y_poly_val_wave < size_y_image
%     for k = size_y_poly_val_wave:size_y_image
%         poly_val_wave(1,k) = poly_val_wave(size_y_poly_val_wave);
%     end
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% obtaining reference pt. for XDN image - not required - covered before
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% max_val = 0;
% size_wave = size(wave_plot);
% size_y_wave = size_wave(2);
% for j = 1:size_y_wave
%     if unmax_poly_val_wave(1,j) > max_val
%         max_val = unmax_poly_val_wave(1,j);
%         reference_j = j;
%     end
% end
% for j = 1:size_y_wave
%     poly_val_wave(1,j) = max_val/unmax_poly_val_wave(1,j);
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% obtaining XDN image : simply multiply
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
size_image       = size(image);
size_y_image     = size_image(2);
size_y_poly_wave = size(poly_val_wave);
for i=1:size_x_image
    y = 1;
    for j = 1:size_y_image-1   
        magnification_factor = poly_val_wave(1,j);
        j_value  = ceil(magnification_factor*j)+j;
        for y = y:j_value
            xdn_image(i,y) = image(i,j); 
        end
        y = j_value + 1;
    end 
end
figure('Name','xdn image'); imshow(xdn_image);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% obtaining second level of transformation: (i) X in terms of X' 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for X = 1:size_y_image - 1
    Xprime(1,X) = ceil(poly_val_wave(1,X)*X)+X;
    vector(1,Xprime(1,X)) = X;
end
half_vector = vector;

size_vector   = size(vector);
size_y_vector = size_vector(2);
count         = 0;
prev_value    = 0;
for j = 1:size_y_vector
    if vector(1,j) == 0
        count = count + 1;
    else
        if count > 0
            if prev_value > 0
                for k = 1:count
                    vector(1,j-count+k-1) = (k/(count+1))*(vector(1,j) - prev_value) + prev_value;
                end
                
            else
                for k = 1:count
                    vector(1,j-count+k-1) = (k/(count+1))*(vector(1,j));
                end
            end
            count = 0;
        end
        prev_value = vector(1,j);
    end
end
y_axis = 1:1:size_y_vector;
figure('Name','Plot of vector relating X to Xprime');plot(y_axis,vector,'b-') ;
%% it comes out to be already smoothed by operations above.
%%poly_vector           = polyfit(y_axis,vector,2); 
%%poly_val_vector       = polyval(poly_vector,y_axis);
%%figure('Name','Plot of smoothed vector relating X to Xprime');plot(y_axis,poly_val_vector,'b-') ;
            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% obtaining second level of transformation: (ii) operation of interpolated
%% average_wave over curve obtained above to get (2f - D*(X')) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%earlier_y_axis = 1:1:size_y_image;
current_y_axis = 1:1:size_y_vector;
%interp_poly_val_wave = interp1(earlier_y_axis, poly_val_wave, current_y_axis,'linear');

%obtain 2f - D*(X')
size_vector   = size(vector);
size_y_vector = size_vector(2);
for Xprime = 1:size_y_vector
    %inverse_poly_val_wave_value = inverse_poly_val_wave(1,Xprime);
    %poly_val_vector_value = poly_val_vector(1,Xprime);
    %sub_Dstar_Xprime(1,Xprime) = 1/poly_val_wave(1, poly_val_vector_value);
    
    vector_value = ceil(vector(1,Xprime));
    %taking an inverse would have same affect as subtracting
    %that.. but why you would take the inverse in the first 
    %place? k.. that was being done because it were not considered
    %appropriate to expand characters in x direction where things
    %were already relevantly clearly seen. An inverse with zero as
    %a value would create unnecessary spikes. So a modification in 
    %Y direction could be sufficiently seen if that is the case. 
    Dstar_Xprime(1,Xprime) = 1-(poly_val_wave(1, vector_value));
end

figure('Name','Plot of second level of transformation wave');plot(current_y_axis, Dstar_Xprime,'b-') ;
min_Dstar_Xprime  = min(Dstar_Xprime);
size_Dstar_Xprime = size(Dstar_Xprime);
size_y_Dstar_Xprime = size_Dstar_Xprime(2);
for j = 1:size_y_Dstar_Xprime
    Dstar_Xprime(1,j)     = Dstar_Xprime(1,j)-min_Dstar_Xprime;
end
figure('Name','Plot of normalized to zero second_level_of_transformation wave');plot(current_y_axis,Dstar_Xprime,'b-');
%inverse_poly_wave     = polyfit(current_y_axis,Dstar_Xprime,2); 
%inverse_poly_val_wave = polyval(inverse_poly_wave,current_y_axis);
%figure('Name','Plot of smoothed vector relating X to Xprime');plot(y_axis,inverse_poly_val_wave,'b-') ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% obtaining second level of transformation: (ii) operation to find t at 
%% which the integral sum of 2f - D*(X') = X'.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% differentiate
%% square + 1
%% square_root 
%% trapz to the point to get first X'. 
%% store the curve(1,step) and get the xfn image in a similar manner

size_dstar = size(Dstar_Xprime);
size_y_dstar = size_dstar(2);
curve_found  = 0;
for j = 1:size_y_dstar - 1
    diff_Dstar_Xprime(1,j)    = Dstar_Xprime(1,j+1) - Dstar_Xprime(1,j);
    sq_diff_Dstar_Xprime(1,j) = diff_Dstar_Xprime(1,j)*diff_Dstar_Xprime(1,j);
    curve_length_part(1,j)    = sqrt(1 + sq_diff_Dstar_Xprime(1,j));
    if curve_length_part(1,j) > 1
        curve_found = 1;
    end
end
size_curve_part = size(curve_length_part);
size_y_curve_part = size_curve_part(2);
y_axis = 1:1:size_y_curve_part;
figure('Name','Plot pertaining to differences in length of directrix');plot(y_axis, curve_length_part,'b-') ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Obtaining the final X coordinates.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
prev_step = 0;
for Xprime = 1:size_y_curve_part - 1
    for final_j_value = 1:size_y_curve_part - 1
        if prev_step > final_j_value
            continue;
        end
        integral_value = trapz(curve_length_part(1,1:final_j_value+1));
        if integral_value > Xprime || integral_value == Xprime
            curve_length(1,Xprime) = final_j_value;
            prev_step = final_j_value;
            break;
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% XFE Elimination
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

size_curve_length = size(curve_length);
size_xdn_image = size(xdn_image);
size_x_xdn_image = size_xdn_image(1);
size_y_curve_length = size_curve_length(2);

for i = 1:size_x_xdn_image
    y = 1;
    for j = 1:size_y_curve_length   
        j_value  = curve_length(1,j);
        for y = y:j_value
            xfe_xdn_image(i,y) = xdn_image(i,j); 
        end
        y = j_value + 1;
    end 
end
figure('Name','xfe xdn image'); imshow(xfe_xdn_image);
                
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% YDN Elimination
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

size_xfe_xdn_image = size(xfe_xdn_image);
size_x_xfe_xdn_image = size_xfe_xdn_image(1);
size_y_xfe_xdn_image = size_xfe_xdn_image(2);
size_dstar_xprime    = size(Dstar_Xprime);
size_y_dstar_xprime  = size_dstar_xprime(2);

for j = 1:size_y_xfe_xdn_image
    x = 1;
    for i = 1:size_x_xfe_xdn_image
        if i > size_y_dstar_xprime 
            i_value = i-ceil((Dstar_Xprime(1,size_y_dstar_xprime))*i);
        else
            i_value  = i-ceil((Dstar_Xprime(1,i))*i);
        end
        for x = x:i_value
            ydn_xfe_xdn_image(x,j) = xfe_xdn_image(i,j); 
        end
        x = i_value + 1;
    end 
end
figure('Name','ydn xfe xdn image'); imshow(ydn_xfe_xdn_image);    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
imtool(ydn_xfe_xdn_image);
imtool(image);
