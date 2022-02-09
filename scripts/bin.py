      

     '''
     # set up data
      #stratify not needed -> data is balanced
     
     
     labels_filtered = filtered_dataset_full['label']

     event_id = dict(relax=1, MI=2)
     #events =  filtered_dataset_full['label']
     #filtered_dataset_full.drop('label', axis=1, inplace=True)
     #print(filtered_dataset_full.columns)
     info = mne.create_info(ch_names = list(filtered_dataset_full.columns), sfreq = sample_duration)
     raw = mne.io.RawArray(filtered_dataset_full.transpose(), info)
     #raw_label = mne.create_info(ch_names = list(labels_filtered.columns), sfreq = sample_duration)
     #raw.add_channels(list(labels_filtered), force_update_info=True)
     matplotlib.use('TkAgg')
     events = mne.find_events(raw, stim_channel='label', initial_event=True)
     #raw.plot(block=True, events=events)

     epochs = mne.Epochs(raw, events, event_id)

     utils.plot_dataset(filtered_dataset_full, ['FZ_','PZ_', 'HL', 'CZ_', 'VD_', 'label'],
                                    ['like', 'like', 'like', 'like', 'like', 'like'],
                                    ['line', 'line', 'line', 'line', 'line', 'line'])




     '''