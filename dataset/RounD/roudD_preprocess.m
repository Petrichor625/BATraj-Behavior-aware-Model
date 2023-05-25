% Preprocess the rounD dataset before the training/evaluation
% A considerable part of this file is inspired by https://github.com/nachiket92/conv-social-pooling/blob/master/preprocess_data.m and https://github.com/m-hasan-n/roundabout

function preprocess_roudD()

%% Parameters

dataset_dir = 'rounD';

%Number of recordings of the dataset to be used
N_records = 22;

%Threshold on acceleration/deceleration
acc_thresh = 0.2;

%consider a vehicle is far from roundabout if exceeding 'dist_factor' multiples of its length
dist_factor = 1;

%History look-back of 2 seconds (2*25 fps = 50)
hist_size = 50;


%% Load the dataset csv files, amend it by the lane IDs, and save into mat files

save_dataset_mat(dataset_dir);

%% Fields:
% 1 recordingId 
% ds_ind = 1;
% 2 trackId  
veh_ind = 2;
% 3 frameId
frame_ind = 3;
% 4 xCenter      
% 5 yCenter       
% 6 heading  
% 7 lonVelocity    
% 8 latVelocity
% 9 lonAcceleration   
% 10 latAcceleration 
% 11 LaneId
lane_ind = 11;
% ==================
% 12 Long-term Vehicle Trajectory class 
traj_class_ind = 12;
% 13 Short-term Lateral Intention
lat_class_ind = 13;
% 14 Longitudinal Intention
lon_class_ind = 14;
%short-term goal whether to stay inside or exit the roundabout
stay_exit_ind = 15;
% 15~ Nbr IDs
nbr_ind_st = 16;

%% Load the refined dataset
traj = cell(N_records,1);
vehTrajs = cell(N_records,1);

for record_no = 2 : 23
    
    %load the refined data
    fname = fullfile(dataset_dir, [sprintf('%02d',record_no) '_tracks.mat']);
    load(fname,'record_data')
    
    %refine the lane IDs
    refined_data = refine_rounD(record_no, record_data, dist_factor);
    
    %normalize the heading to be in rad instead of deg
    refined_data(:,6) = refined_data(:,6)*pi/180;
    
    % adjust record Ids to be from 1:22 instead of 2:23
    refined_data(:,1) = refined_data(:,1) - 1;
    
    %store
    traj{record_no-1} = single(refined_data);
    vehTrajs{record_no-1} = containers.Map;
   
   
end

%% Parse the fields
for ii = 1 : N_records
     
    %Unique Vehicle IDs 
     vehIds = unique(traj{ii}(:,veh_ind));
     for v = 1:length(vehIds)
         veh_traj = traj{ii}(traj{ii}(:,veh_ind) == vehIds(v),:);
         vehTrajs{ii}(int2str(vehIds(v))) = veh_traj;
         
         %Find the long-term goal class of each vehicle 
         %-----------------------------------------------------------
         %This is based on the entry and exit lanes
         entry_lane = veh_traj(1, lane_ind);
         exit_lane = veh_traj(end, lane_ind);
         trj_class = find_trj_class(entry_lane, exit_lane);
         traj{ii}(traj{ii}(:,veh_ind) == vehIds(v), traj_class_ind) = trj_class;
         
         %Find the short-term Lateral intention
         %-----------------------------------------------
         %Ids of this vehicle in the traj subset
         traj_ids = find(traj{ii}(:,veh_ind) == vehIds(v));
         %Which lanes are traversed by this vehicle 
         traversed_lanes= veh_traj(:, lane_ind);
         %Times at which a lane change happens
         change_times = find(diff(traversed_lanes));
         %Iterate on the lane change time and set the lateral intention
         for kk =1:length(change_times)
             %what is the next lane that changed to
             fut_lane_id = change_times(kk)+1;
             
             %Intention is based on the next lane
             [lat_class, entry_exit_class] = find_lat_int(traversed_lanes(fut_lane_id));
             
             %Indices to set the lateral intention class   
             if kk==1
                st = 1;
             else
                 st = change_times(kk-1) + 1;
             end
             if kk==length(change_times)
                en = size(traversed_lanes,1);
             else
                 en = change_times(kk);
             end
             %Set the lateral intention class 
             traj{ii}(traj_ids(st):traj_ids(en), lat_class_ind) = lat_class;  
             traj{ii}(traj_ids(st):traj_ids(en), stay_exit_ind) = entry_exit_class;
         end
         
         %Find the Longitudinal intention
         %---------------------------------------
         lonAcc = veh_traj(:,9);
         %Initialize the Lon intention with ones (normal class)
         lon_int_class = ones(size(veh_traj,1),1);
         %deceleration class
         lon_int_class( lonAcc < -acc_thresh ) = 2;
         %acceleration class
         lon_int_class( lonAcc > acc_thresh ) = 3;
         
         %Set the longitudinal intention class 
         traj{ii}(traj_ids, lon_class_ind) = lon_int_class; 
         
     end
        
     
     %Iterate on each timestep
     for k = 1:length(traj{ii}(:,1))    
         
         %dsId = traj{ii}(k, ds_ind);
         vehId = traj{ii}(k, veh_ind);
         time = traj{ii}(k, frame_ind);
         
         laneId = traj{ii}(k, lane_ind);
         
         % Get Nbr Vehicle IDs
         % This will also depend on the lane ID (entry-exit-inside)
         nbr_vehicle_ids = find_nbrs(laneId, vehId, time,traj{ii});
         traj{ii}(k, nbr_ind_st: nbr_ind_st+length(nbr_vehicle_ids)-1) = nbr_vehicle_ids;
        
     end
    
end

%% Split train, validation, test

trj_sizes=zeros(N_records,1);
for ii = 1 : N_records
    trj_sizes(ii)= size(traj{ii},2); 
end
max_siz = max(trj_sizes);

%adjust the size of traj cell array 
trajAll = [];
for ii = 1 : N_records
    trajAll = [trajAll;  [traj{ii} zeros(size(traj{ii},1),  max_siz- trj_sizes(ii)) ] ];
end
% clear traj;

trajTr = [];
trajVal = [];
trajTs = [];

%Iterate on the record Ids from 2 to 23
for k = 1:N_records
    uniq_veh = unique(trajAll(trajAll(:,1)==k,2));
    n_uniq_veh = length(uniq_veh);
    lim1 = uniq_veh(round(0.7*n_uniq_veh));
    lim2 = uniq_veh(round(0.8*n_uniq_veh));
   
    trajTr = [trajTr;trajAll(trajAll(:,1)==k & trajAll(:,2)<=lim1, :)];
    trajVal = [trajVal;trajAll(trajAll(:,1)==k & trajAll(:,2)>lim1 & trajAll(:,2)<=lim2, :)];
    trajTs = [trajTs;trajAll(trajAll(:,1)==k & trajAll(:,2)>lim2, :)];
end

%Tracks of each vehicle
%[frame_id X Y Heading]
tracksTr = {};
for k = 1:N_records
    trajSet = trajTr(trajTr(:,1)==k,:);
    carIds = unique(trajSet(:,2));
    for l = 1:length(carIds)
        vehtrack = trajSet(trajSet(:,2) ==carIds(l),3:6)'; 
        tracksTr{k,carIds(l)} = vehtrack;
    end
end

tracksVal = {};
for k = 1:N_records
    trajSet = trajVal(trajVal(:,1)==k,:);
    carIds = unique(trajSet(:,2));
    for l = 1:length(carIds)
        vehtrack = trajSet(trajSet(:,2) ==carIds(l),3:6)';
        tracksVal{k,carIds(l)} = vehtrack;
    end
end

tracksTs = {};
for k = 1:N_records
    trajSet = trajTs(trajTs(:,1)==k,:);
    carIds = unique(trajSet(:,2));
    for l = 1:length(carIds)
        vehtrack = trajSet(trajSet(:,2) ==carIds(l),3:6)';
        tracksTs{k,carIds(l)} = vehtrack;
    end
end


%% Filter edge cases: 
% Since the model uses 2 sec of trajectory history for prediction, 
% the initial 2 seconds of each trajectory is not used for training/testing

indsTr = zeros(size(trajTr,1),1);
for k = 1: size(trajTr,1)
    t = trajTr(k,3);
    t_track = tracksTr{trajTr(k,1),trajTr(k,2)};
    if size(t_track,2)>hist_size
        if t_track(1,hist_size+1) <= t && t_track(1,end)>t+1
            indsTr(k) = 1;
        end
    end
end
trajTr_full = trajTr;
trajTr = trajTr(find(indsTr),:);
 
indsVal = zeros(size(trajVal,1),1);
for k = 1: size(trajVal,1)
    t = trajVal(k,3);
    t_track = tracksVal{trajVal(k,1),trajVal(k,2)};
    if size(t_track,2)>hist_size
        if t_track(1,hist_size+1) <= t && t_track(1,end)>t+1
            indsVal(k) = 1;
        end
    end
end
trajVal_full = trajVal;
trajVal = trajVal(find(indsVal),:);

indsTs = zeros(size(trajTs,1),1);
for k = 1: size(trajTs,1)
    t = trajTs(k,3);
    t_track = tracksTs{trajTs(k,1),trajTs(k,2)};
    if size(t_track,2)>hist_size
        if t_track(1,hist_size+1) <= t && t_track(1,end)>t+1
            indsTs(k) = 1;
        end
    end
end
trajTs_full = trajTs;
trajTs = trajTs(find(indsTs),:);

%% Amend the dataset with the anchor trajectories

[tr_anchors, val_anchors, ts_anchors, anchor_traj_raw] = ....
    trj_anchors_amend_dataset(trajTr,tracksTr,trajTr_full,...
        trajVal,tracksVal,trajVal_full,...
            trajTs, tracksTs,trajTs_full);

%% Save mat files:
traj = trajTr;
tracks = tracksTr;
traj_full = trajTr_full;
tracks_anchored = tr_anchors;
save('TrainSet','traj','tracks','traj_full','tracks_anchored','anchor_traj_raw');

traj = trajVal;
tracks = tracksVal;
traj_full = trajVal_full;
tracks_anchored = val_anchors;
save('ValSet','traj','tracks','traj_full','tracks_anchored','anchor_traj_raw');

traj = trajTs;
tracks = tracksTs;
traj_full = trajTs_full;
tracks_anchored = ts_anchors;
save('TestSet','traj','tracks','traj_full','tracks_anchored','anchor_traj_raw');

end

%%  Helper Functions
%
%
%

%distance_to_roundabout function computes the distance between the vehicle at any timestep
%and the conflict area of the roundabout
function dist_round = distance_to_roundabout(veh_traj, lane_ids)

%Init with zeros: if the vehicle is inside the roundabout conflict area, 
%then dist_round = 0, same as the initialized value
dist_round = zeros(size(veh_traj,1), 1);

%entry and exit lane IDs
entry_exit_lanes =[1, 2, 3, 4, 6, 7, 9, 10, 12, 13, 14, 15]';

%Distance is computed from these limit points that wrere found manually
entry_exit_lane_points = [73.4, -70.3;
78.1, -71.3;
106.4, -51.2;
107.1, -46.8;
90.2, -22.9;
85.3, -21.8;
57.5, -42.3;
57.1, -47.5;
66.2, -66;
103, -60;
98.6, -27.7;
59.9, -35];

%Iterate on all timesteps of the trajectory 
for ii = 1 : size(veh_traj, 1)
    
    lane_id = lane_ids(ii);
    
    %assign -1 value for vehicles in the non-relevant lanes
    if  any(ismember(lane_id,[5,8,11,200]))
        dist_round(ii) = -1;
        
        %Entry/Exit lanes
    elseif any(ismember(lane_id, entry_exit_lanes))
        ind = entry_exit_lanes==lane_id;
        lane_point = entry_exit_lane_points(ind,:);
        dist_round(ii) = (sum((veh_traj(ii,:) - lane_point).^2))^0.5;
    end
    
end

end


%% Future short-term lateral intention is defined by intended the lane
function [lat_int , lat_goal ]= find_lat_int(next_lane)

if ismember (next_lane,[106,116,1, 2])
    lat_int = 1;
elseif ismember (next_lane,[107, 117,13])
    lat_int = 2;
elseif ismember (next_lane,[108, 118, 3, 4])
    lat_int = 3;
elseif ismember (next_lane,[101, 111,14])
    lat_int = 4;
elseif ismember (next_lane,[6, 7, 102, 112])
    lat_int = 5;
elseif ismember (next_lane,[103, 113,15])
    lat_int = 6;
elseif ismember (next_lane,[9, 10, 104, 114])
    lat_int = 7;
elseif ismember (next_lane,[12,105, 115])
    lat_int = 8;
else
    lat_int = 0;
end

%Stay inside or Exit the roundabout
if ismember (next_lane,[12,13,14,15])
    lat_goal = 1;
else
    lat_goal = 0;
end

end

%% This is based on 4*4=16 clusters
function anchor_id = find_trj_class(entry_lane, exit_lane)

if (entry_lane==1 || entry_lane==2)
    entry_id =1;
elseif (entry_lane==3 || entry_lane==4)
    entry_id = 2;
elseif (entry_lane==6 || entry_lane==7)
    entry_id = 3;
elseif (entry_lane==9 || entry_lane==10)
    entry_id = 4;   
else
    entry_id = 0;
end

%exit lanes are from 12 to 15
if entry_id ~=0 && ismember(exit_lane,12:15) 
    exit_id = exit_lane-11;
    anchor_id = (entry_id-1)*4 + exit_id;
else
    anchor_id = 0;
end

end

%% Find the Nbr vehicles
function nbr_vehicles = find_nbrs(laneId, vehId, frameId, traj)

%find all vehicles at this frame
frame_inds = traj(:,3)==frameId;
same_frame_data = traj(frame_inds,:);

%find the lanes traversed by these vehicles
lanes = same_frame_data(:,11);

%If the ego has already exit the conflict zone
%Then consider only the nbrs at the same exit lane
if ismember(laneId,[12 13 14 15])
    considered_lanes = lanes==laneId;

else
    %which of these lanes are inside the conflict zone
    considered_lanes = lanes >100;

    %find the neighboring lane to that of the Ego 
    entry_exit_lanes = [1 2 3 4 6 7 9 10 12 13 14 15];
    lane_indx = entry_exit_lanes==laneId;
    nbr_lanes = [2 1 4 3 7 6 10 9 12 13 14 15];
    nbr_lane = nbr_lanes(lane_indx);

    %we consider lanes inside the conflict zone and the neighboring lane 
    %to that of the ego
    if ~isempty(nbr_lane)
        considered_lanes = considered_lanes | lanes== nbr_lane;
    end    
end

%Nbr Vehicle IDs
nbr_vehicles = same_frame_data(considered_lanes,2);
nbr_vehicles = setdiff(nbr_vehicles,vehId);
end

%% load the datset csv files and save them into mat
function save_dataset_mat(dataset_dir)

% Load roundabout and lane boundaries
% fname = fullfile(dataset_dir, 'lanes\lane_limits.mat');
% load(fname,'lane_ids','lane_boundaries')

%% Vehicle Class Names
% List of classes: 
class_names = {'car', 'truck', 'bus', 'van', 'trailer', 'pedestrian', 'bicycle', 'motorcycle'};

%% Iterate on all recordings 
% exclusing records 0 and 1 as they come from different locations
% recorde 2~23 all come from the same roundabout environment
for record_no = 2 : 23
    
    %% load the tracks metadata
    fname = fullfile(dataset_dir, [sprintf('%02d',record_no) '_tracksMeta.csv']);
    tracks_meta = readtable(fname);
    
    %% load the tracks table
    % 1 recordingId       2 trackId       3 frame     4 trackLifetime
    % 5 xCenter      6 yCenter       7 heading   8 width     9 length
    % 10 xVelocity   11 yVelocity    12 xAcceleration    13 yAcceleration
    % 14 lonVelocity    15 latVelocity      16 lonAcceleration   17 latAcceleration
    fname = fullfile(dataset_dir, [sprintf('%02d',record_no) '_tracks.csv']);
    record_table = table2array(readtable(fname));
    
    %Init the amended table with the availabe data
    record_data =record_table;
    
    %% lane IDs
    fname = fullfile(dataset_dir, [sprintf('%02d',record_no) '_laneIDs.txt']);
    record_lane_ids = table2array(readtable(fname));
    
    %% Iterate on Vehicles in this record
    % to amend the data with lane ID, distance to roundabout, class id
    unique_veh_ids = unique(record_table(:,2));
    
    for ii = 1: length(unique_veh_ids)
        
        veh_id = unique_veh_ids(ii);
        
        veh_inds = record_table(:,2) == veh_id;
        no_frames = sum(veh_inds);
        
        veh_class = table2cell(tracks_meta(ii,8));
        class_id = find(strcmp(class_names, veh_class))*ones(no_frames,1);
        veh_lane_ids = record_lane_ids(veh_inds);  
                
        veh_data = record_table(veh_inds,:);
        veh_traj = veh_data(:,5:6);
        dist_round = distance_to_roundabout(veh_traj, veh_lane_ids);
        
        %18 Class ID    19 Lane ID   20 Dist_roundabout
        record_data(veh_inds,18:20) = [class_id veh_lane_ids dist_round];

    end
    
    %Saving
    fname = fullfile(dataset_dir, [sprintf('%02d',record_no) '_tracks.mat']);
    save(fname,'record_data')

end

end


%% Refine the dataset by excluding non-relevant data
function refined_data = refine_rounD( record_id,old_data,dist_factor)
%% Fields to refine
% 1 recordingId       2 trackId       3 frame     4 trackLifetime
% 5 xCenter      6 yCenter       7 heading   8 width     9 length
% 10 xVelocity   11 yVelocity    12 xAcceleration    13 yAcceleration
% 14 lonVelocity    15 latVelocity      16 lonAcceleration   17 latAcceleration
% 18 Class    19 Lane     20 Dist_roundabout

%% exclude pedestrian, bicycle and motorcycle classes
% class_names = {'car', 'truck', 'bus', 'van', 'trailer', 'pedestrian', 'bicycle', 'motorcycle'};
class_ids = old_data(:,18);
class_refine = class_ids==6 | class_ids==7 | class_ids==8 ;

%% exclude trajectories in none-relevant lanes 5,8,11,200
lane_ids = old_data(:,19);
lane_refine = lane_ids==5 | lane_ids==8 | lane_ids==11 | lane_ids==200;

%% exclude trajectory points that are far from roundabout (by a threshold)
%threshold equals a dist_factor multiples of vehicle length
dist_roundabout =  old_data(:,20);
dist_threshold = dist_factor*old_data(:,9);
dist_refine = dist_roundabout > dist_threshold;

refine_ids = class_refine | lane_refine | dist_refine;

refined_data = old_data;
refined_data(refine_ids,:) = [];

%% Remove non-relevant columns and keep the following
% 1 recordingId       2 trackId       3 frame
% 5 xCenter      6 yCenter       7 heading     
% 14 lonVelocity    15 latVelocity      16 lonAcceleration   17 latAcceleration 
%19 Lane  
 refined_data = refined_data(:,[1 2 3 5 6 7 14 15 16 17 19]);
 
%%  Refine the lane IDs
refined_data = refine_lanes(record_id, refined_data);
end


%% refine lane IDs
function refined_data = refine_lanes(record_id, refined_data)

    
%In the following, set the first IDs as laneX insted of the given id
%[record_id veh_id given_lane_id new_lane_id]
entry_lane_corr =    [2 3 10
                                 3  4  103
                                 3  6   103
                                 3  490   2
                                 4    1         3
                                 7    261   2
                                 7  356 7
                                 7  386 10
                                 8  157  2
                                 8  226     7
                                 8      384     10
                                 9      380     10
                                 12     8       7
                                 12     247   4
                                 14     1       103
                                 14     2       103
                                 14     4       107
                                 17     26      10
                                 18     482     10
                                 18     488     10
                                 19     2         10
                                 20     180     10
                                 21     2       10
                                 23     171     4];

exit_lane_corr=[3     546          12 
                         3     564          12
                         3      499         12
                         5      406         12
                         6      153         13
                         8      339         12
                         9      200         12
                         9      703         15
                         ];
                          
%Delete trajectories of the following Vehicle Ids because they are too short
delete_traj_short{2}=[12 325 ];
delete_traj_short{5}= 12;
delete_traj_short{7} = 632;
delete_traj_short{8} = 509;
delete_traj_short{10} = 1 ;
delete_traj_short{11} = [2  5  542];
delete_traj_short{12} = 514 ;
delete_traj_short{13} = [293    294 295] ;
delete_traj_short{14} = [3  5   548     564];
delete_traj_short{18} = 693;
delete_traj_short{19} = [487    488];
delete_traj_short{20} = 5;
delete_traj_short{21} = [304    708 709];
delete_traj_short{22} = 3;
delete_traj_short{23} = [6  8   11  353 449 454];

%Delete trajectories of the following Vehicle Ids because they exist in 
% lane "200a"
delete_traj_200{4} = 418;
delete_traj_200{6} = [355   518];
delete_traj_200{7} = [487   532];
delete_traj_200{9} = [28    61  74  138 209 215 216 227 272 278 301 330 ...
                                  333   361  403    445 494 529 537 588 593 618 632 634 ...
                                  687   689 690 692 ];

delete_traj_200{10} = [11   33  43  63  116 134 159 166 222 224 239 273 316 ...
                                    384 385 415 443 497 501 545 556 663 ];

delete_traj_200{11} = [102  118   142     261   277 302 310 315 317 379 380 ...
                                    382 463 467 484     500     523     541];

delete_traj_200{12} = [16   66  77  140 171 177 179 183 240 256 266 299 ...
                                     366 374 392 405 444 509 512 ];

delete_traj_200{13} = [   89  146     153     176     178     196     207 ...
                                    267     270];

delete_traj_200{14} = [61   74  77  82  104 121 165 180 187 199     221     224 ...
                                    242 264 268 310 313 399 418 492 496 536 549];

delete_traj_200{15} = [40   64  94  181 184];

delete_traj_200{16} = [17   21  79  81  89  120 133 139 175 179 197 263];

delete_traj_200{17} = [40   74  174 185 200 210 237 256 327 378 478 492 ...
                                     509    510 517 555 563 572 594 597 608 674];

delete_traj_200{18} = [38   44  48  96  126 149 186 187 207 282 286 302 316 ...
                                    377 380 396 545 550 586 589 640 643 665 668]; 

delete_traj_200{19} = [31   43  50  73  118 206 218 242 243 294 319 324 ...
                                    337 399 404 430 467];      
                                
delete_traj_200{20} = [67   141 143 181 223 256 269 314 345 352 410 456 ...
                                    518 520 541 582 690 696 746 750];
                                
delete_traj_200{21} =[42    45  94  98  152 155 158 159 162 165 167 177 266 ...
                                   314  316 320 366 380 387 415 441 496 505 548 549 ...
                                   613  632 659 700];
                               
delete_traj_200{22} =[11    70  146 157 173 195 220 298 334 384 387 393 422 ...
                                    427 464 470 472 473 485 516 533 568];
                                
delete_traj_200{23} =[31    62  63  138 184 199 225 272 290 301 309 320 366 ...
                                   378  417];

%Delete trajectories of the following Vehicle Ids because they exist in
%lane "8"
delete_traj_8{9} = [158 178     212 279 284     309   326   514 550 551  604];

delete_traj_8{10} = [19 31  120 125 133 180 258 288 350 375 421 481  ...
                                498 523 576 609 627 628 645 ];

delete_traj_8{11} = [15 21  30  31  42  61  111 126   134   161     186     243 ...
                                245   299    308 473    485];

delete_traj_8{12} = [23 43  142 176 188 232 278 354 355 399 423];                            

delete_traj_8{13} = [72 123    142     203     217];

delete_traj_8{14} = [29 49  149 162 222 226 295 347 391 431 433 527];

delete_traj_8{15} = [31 59  102 104 114 210 219 220];

delete_traj_8{16} = [31 45];

delete_traj_8{17} = [21 31  194 213 214 290 306 308 332 335 341 407 464 474 ...
                                505 515 536 587 616 652 667 669];

delete_traj_8{18} = [49 70  104 134 170 216    329 342 408 459 531 677 678];

delete_traj_8{19} = [11 56  68  125 164 207 211 222 249 276 279 282 321 342 ...
                                361 367 373 418 435];

delete_traj_8{20} =[214 325 495 536 553 594 714 753];

delete_traj_8{21} =[86  113 150 151 173 228 232 292 344 368 372 384 402 501 ...
                                642 643 652 665];
                            
delete_traj_8{22} =[135 171 179 213 229 376 406 438 479 531 567];                            

delete_traj_8{23} =[28  49  75  116 154 234 244 261 268 328 338 371 389 398 448];
                       
vehIds = unique(refined_data(:,2));


%% correct entry-lane ids
ent_ids = entry_lane_corr(entry_lane_corr(:,1)== record_id,2);
ent_ids = vehIds(ent_ids);
ent_corr_lanes = entry_lane_corr(entry_lane_corr(:,1)== record_id,3);

for ii = 1 : length(ent_ids)
    
    inds = refined_data(:,2)==ent_ids(ii);
    lane_ids = refined_data(inds,11);
    zero_ids = lane_ids==0;
    
    if any(zero_ids) 
        if lane_ids(1)== 0
            k = find(inds);
            refined_data(k(zero_ids),11) = ent_corr_lanes(ii);
        else
            pause
        end
    else
        k = find(inds);
        refined_data(k(1),11) = ent_corr_lanes(ii);
    end
    
end

%% correct exit-lane ids
exit_ids = exit_lane_corr(exit_lane_corr(:,1)== record_id,2);
exit_ids = vehIds(exit_ids);
exit_corr_lanes = exit_lane_corr(exit_lane_corr(:,1)== record_id,3);

for ii = 1 : length(exit_ids)
    
    inds = refined_data(:,2)==exit_ids(ii);
    lane_ids = refined_data(inds,11);
    zero_ids = lane_ids==0;
    
    if any(zero_ids) 
        if lane_ids(end)== 0
            k = find(inds);
            refined_data(k(zero_ids),11) = ent_corr_lanes(ii);
        else
            pause
        end
    else
        k = find(inds);
        refined_data(k(end),11) = exit_corr_lanes(ii);
    end
end


%short traj
traj_ids = delete_traj_short{record_id};
short_traj_ids = vehIds(traj_ids);

%lane 200a
traj_ids = delete_traj_200{record_id};
traj_ids_200 = vehIds(traj_ids);

%lane 8
traj_ids = delete_traj_8{record_id};
traj_ids_8 = vehIds(traj_ids);

delete_list = [short_traj_ids;traj_ids_200;traj_ids_8];
for ii = 1 : length(delete_list)
    inds = refined_data(:,2)==delete_list(ii);
    refined_data(inds,:)=[];
end
end

%% anchor trajectories
function [tr_anchors, val_anchors, ts_anchors, anchor_traj_raw] = ...
trj_anchors_amend_dataset(trajTr,tracksTr,trajTr_full,...
    trajVal,tracksVal,trajVal_full,...
    trajTs, tracksTs,trajTs_full)

load('anchor_trajectories','anchor_traj_raw')

tr_anchors = find_anchor_tracks(trajTr,trajTr_full,anchor_traj_raw,size(tracksTr));

val_anchors = find_anchor_tracks(trajVal,trajVal_full,anchor_traj_raw,size(tracksVal));

ts_anchors = find_anchor_tracks(trajTs,trajTs_full,anchor_traj_raw,size(tracksTs));

end

function tracks_anchored = find_anchor_tracks(traj,traj_full,anchor_traj_raw,tracks_size)

veh_ind = 2;
lat_class_ind = 13;
lon_class_ind = 14;
N_records = 22;

tracks_anchored = cell(tracks_size);

for k = 1:N_records
    trajSet = traj_full(traj_full(:,1)==k,:);
    trajSet_r = traj(traj(:,1)==k,:);
    vehIds = unique(trajSet_r(:,veh_ind));
    for v = 1:length(vehIds)
        veh_traj = trajSet(trajSet(:,veh_ind) == vehIds(v),:);
%         veh_traj = tracks{k,v}';
        veh_traj_new = veh_traj;
       
        lat_int = veh_traj(:,lat_class_ind);
        lon_int = veh_traj(:,lon_class_ind);
        
        lat_change_times = find(diff(lat_int));
        lon_change_times = find(diff(lon_int));
        change_times = union(lat_change_times,lon_change_times);
        if size(change_times,1)==1
            change_times = change_times';
        end
        change_times = [change_times;size(veh_traj,1)];
        
        for ii = 1 : length(change_times)
            
            if ii==1
                st =1;
            else
                st = change_times(ii-1)+1;
            end
            
            en = change_times(ii);
            
            
            assert(sum(diff(lat_int(st:en))) ==0);
            assert(sum(diff(lon_int(st:en))) ==0);
            
            trj_seg = veh_traj(st:en,4:6);
            lat_id = veh_traj(st:en,lat_class_ind);
            lon_id = veh_traj(st:en,lon_class_ind);
            
            if lat_id(1)==0 || lon_id(1)==0
                continue
            else
                trj_anch = anchor_traj_raw{lon_id(1) , lat_id(1)};
            end
            
            %deviation from the anchor trajectory
            trj_anch_dev = find_anchor_seg(trj_anch,trj_seg);
            veh_traj_new(st:en,4:6) = trj_anch_dev;
        end
        
        tracks_anchored{k, vehIds(v)} = veh_traj_new(:,3:6)';
        
    end

    
end

end

function trj_anch_dev = find_anchor_seg(trj_anch,trj_seg)

ref_pos = trj_anch(1,:) - trj_seg(1,:);

%Find the segemnt of the anchor trajectory that is most similar to trj_seg
st = trj_seg(1,:);
en = trj_seg(end,:);
st_dist = sum((trj_anch-st).^2,2);
en_dist = sum((trj_anch-en).^2,2);
[~, st_id] = min(st_dist);
[~, en_id] = min(en_dist);

if (st_id==en_id) && (size(trj_seg,1)>1)
    [~,is] = sort(en_dist);
    en_id = is(2);
end

%flip if required
if st_id<=en_id
    trj_anch_seg = trj_anch(st_id:en_id,:);
else
   trj_anch_seg = trj_anch(en_id:st_id,:);
   trj_anch_seg = flip(trj_anch_seg);
end 


n_trj_seg = size(trj_seg,1);
n_anch_seg =   size(trj_anch_seg,1);
if n_trj_seg > n_anch_seg
    %interpolate to make both segments at the same size 
    %(increase the anchor segment no. of timestamps)
    inds = round(linspace(1,n_trj_seg,n_anch_seg));
    inds_q = 1:n_trj_seg;
    Xq = interp1(inds,trj_anch_seg(:,1),inds_q);
    Yq = interp1(inds,trj_anch_seg(:,2),inds_q);
    Hq = interp1(inds,trj_anch_seg(:,3),inds_q);
    trj_anch_seg_interp = [Xq' Yq' Hq'];
    
    %deviation from the anchor trajectory
    trj_anch_dev = trj_anch_seg_interp - trj_seg;
    
elseif n_trj_seg < n_anch_seg
    inds = round(linspace(1,n_anch_seg,n_trj_seg));
    trj_anch_seg_sampled= trj_anch_seg(inds,:);
    %deviation from the anchor trajectory
    trj_anch_dev = trj_anch_seg_sampled - trj_seg;
    
else
    trj_anch_dev = trj_anch_seg - trj_seg;
end

trj_anch_dev = trj_anch_dev + ref_pos;

plot(trj_anch(:,1),trj_anch(:,2),'r')
hold on
plot(trj_seg(:,1),trj_seg(:,2),'g')
hold on
plot(trj_anch_seg(:,1),trj_anch_seg(:,2),'b')

% if n_trj_seg > n_anch_seg
%     plot(trj_anch_seg_interp(:,1),trj_anch_seg_interp(:,2),'m--')
% elseif n_trj_seg < n_anch_seg
%     plot(trj_anch_seg_sampled(:,1),trj_anch_seg_sampled(:,2),'m--')
% end
end
