from tensorboard.backend.event_processing import event_accumulator
from pprint import pprint

ea = event_accumulator.EventAccumulator('logs/tsp_100/run_20220720T144358/events.out.tfevents.1658321038.LAPTOP-DAVIDE',
    size_guidance={ # see below regarding this argument
    event_accumulator.COMPRESSED_HISTOGRAMS: 500,
          event_accumulator.IMAGES: 4,
      event_accumulator.AUDIO: 4,
      event_accumulator.SCALARS: 0,
   event_accumulator.HISTOGRAMS: 1,
     })

ea.Reload() # loads events from file

pprint(ea.Tags()['scalars'])

serie = ea.Scalars('avg_cost')
for s in serie:
    pprint(s.value)
