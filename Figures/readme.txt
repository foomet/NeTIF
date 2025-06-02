To generate many of the plots from the paper, run the "paper_plot.py" script.


Discussion on old_figure3:

In our experiments, to determine the value of p, i.e. the number of packets to select from each flow, we plotted
histograms exhibiting the total number of flows containing a certain number of packets in Figure 3. In the x-axis of the
figure, we show up to 15 packets per flow; in the y-axis we show the counts of flows. From Figure 3, we see that the
majority of flows contain only a small number of packets, with 101,807 flows having a single packet in CICIDS2017 and
69,326 in UNSW-NB15, respectively. The number of flows drops sharply as the packet count increases, with very few
flows exceeding 7 packets per flow. Notably, only 212 flows in UNSW-NB15 and 20,538 in CICIDS2017 have 15 or more
packets, indicating a strong skew toward shorter flows. Since we aim to retrieve a fixed number of packets from each flow
and most flows in both datasets contain only one packet, we set the default value of p to 1 in our experiments to ensure
maximum data utilization and consistency across datasets. This choice allows us to retain most flows and avoid discarding
valuable data due to flow length constraints. At the same time, we also evaluate the effect of the
parameter p on the quality of the generated data for supporting anomaly detection, as discussed in Section IV-E4. We partitioned
each flow image collection into training and testing sets using 80% and 20% splitting.
