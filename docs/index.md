---8<--- "README.md:description"


``` mermaid
graph TB
  A01@{ shape: lean-r, label: "gas: CH4 or CO2" }
  A02@{ shape: lean-r, label: "quantile in [0,100]" }
  style A01 fill:#e99830
  style A02 fill:#e99830
  A01 --> ide1
  A02 --> ide1
  A01 --> ide2
  A02 --> ide2
  subgraph ide1 [ground-based data]
    id1[(NOAA)] --> A1[download data]
    id2[(AGAGE)] --> A1[download data]
    id3[(GAGE)] --> A1[download data]
  end
  subgraph ide3 [preprocessing]
    A1 --> B[binning to 5x5 grid]
    B --> C[interpolation]
    C --> D[model vertical]
  end
  subgraph ide2 [earth observations]
    id4[(OBS4MIPS)] --> A2[download data]
  end
  B -.-> B2@{ shape: bow-rect, label: "{gas}_binned.csv" }
  style B2 fill:#cee4d8,stroke:#007a3d
  C -.-> C2@{ shape: bow-rect, label: "{gas}_interpolated.csv" }
  style C2 fill:#cee4d8,stroke:#007a3d
  D -.-> D2@{ shape: bow-rect, label: "{gas}_vertical.csv" }
  style D2 fill:#cee4d8,stroke:#007a3d
  A2 -.-> A3@{ shape: bow-rect, label: "{gas}_raw.csv" }
  style A3 fill:#cee4d8,stroke:#007a3d
  A1 -.-> A4@{ shape: bow-rect, label: "{gas}_raw.csv" }
  style A4 fill:#cee4d8,stroke:#007a3d
  A2 --> E[combine datasets]
  D --> E[combine datasets]
  E --> F[filter according to EO]
  F --> G[apply averaging kernel]
  E -.-> E2@{ shape: bow-rect, label: "{gas}_joint_wide.csv" }
  style E2 fill:#cee4d8,stroke:#007a3d
  G -.-> G2@{ shape: bow-rect, label: "{gas}_joint_comparison.csv" }
  style G2 fill:#cee4d8,stroke:#007a3d
```
