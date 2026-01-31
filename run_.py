# from models.GATEncoder import GATEncoder
# from models.GCNEncoder import GCNEncoder
# from models.RGCNEncoder import RGCNEncoder
# from tdg_bench import TDGBench
# from train.Trainer import Trainer
# from train.semi_supervised_classification.ssc_train import training_loop_minibatch
#
# if __name__ == "__main__":
#     tdg = TDGBench()
#     # tdg.generate_splits()
#     annotated_graph, train_loader, val_loader, test_loader, gdp = tdg.get_data(kg_name = "GT2KG_kg",init_embd="sentence-transformers/all-MiniLM-L6-v2", entities_embd_path = "GT2KG_kg_init_embd/EntitiesEmbedding.pickle", edges_embd_path = "GT2KG_kg_init_embd/PredicatesEmbedding.pickle")
#     print(annotated_graph)
#     in_channels = annotated_graph.x.shape[1]
#     rel = set(annotated_graph.edge_type.tolist())
#     num_relations = len(rel)
#     hidden_channels = 256
#     out_channels = 256
#     # gcn_encoder = GCNEncoder(in_channels = in_channels, hidden_channels = hidden_channels, out_channels = out_channels)
#     # gcn_model = tdg.prepare_model(gcn_encoder)
#     # print(gcn_model)
#     # gat_encoder = GATEncoder(in_channels = in_channels, hidden_channels = hidden_channels, out_channels = out_channels)
#     # gat_model = tdg.prepare_model(gat_encoder)
#     # print(gat_model)
#     RGCN_encoder = RGCNEncoder(in_channels = in_channels, hidden_channels = hidden_channels, out_channels = out_channels, num_relations=num_relations)
#     RGCN_model = tdg.prepare_model(RGCN_encoder)
#     # print(RGCN_model)
#
#     trainer = Trainer(
#         model=RGCN_encoder,
#         device='cuda',
#         lr=0.01,
#         weight_decay=5e-4,
#         optimizer_type='adam'
#     )
#
#     results = trainer.train(
#         train_loader=train_loader,
#         val_loader=val_loader,
#         test_loader=test_loader,
#         epochs=100,
#         patience=100,
#         verbose=True
#     )
#
#     print(f"Best Val F1: {results['best_val']['f1']:.4f}")
#     print(f"Test Accuracy: {results['final_test']['accuracy']:.4f}")
#     print(f"Test F1: {results['final_test']['f1']:.4f}")
#
#     # trainer.evaluate(
#     #     test_loader,
#     #     split='test',
#     #     save_predictions=True,
#     #     save_path='predictions.xlsx',
#     #     label_encoder=gdp.label_encoder,
#     #     decode_indexes_fn=lambda id: gdp.decode_indexes()[id]
#     # )
#
from models.GCNEncoder import GCNEncoder
from models.RGCNEncoder import RGCNEncoder
from tdg_bench import TDGBench

if __name__ == "__main__":
    # Initialize benchmark
    tdg = TDGBench(use_classifier=True)

    # Optionally generate splits if needed
    # tdg.generate_splits()

    # Get data once to extract graph properties
    # annotated_graph, _, _, _, _ = tdg.get_data(
    #     kg_name="reference_kg",
    #     init_embd="sentence-transformers/all-MiniLM-L6-v2",
    #     entities_embd_path="BioMed_noisy_init_embd/EntitiesBertEmbedding_NCI.pickle",
    #     edges_embd_path="BioMed_noisy_init_embd/PredicatesBertEmbedding_NCI.pickle"
    # )
    # tdg.generate_splits()
    annotated_graph, _, _, _, _ = tdg.get_data(
        kg_name="GT2KG_kg",
        init_embd="sentence-transformers/all-MiniLM-L6-v2",
        entities_embd_path = "GT2KG_kg_init_embd/EntitiesEmbedding.pickle",
        edges_embd_path = "GT2KG_kg_init_embd/PredicatesEmbedding.pickle",
        )
    print(annotated_graph)
    # Extract graph properties for model initialization
    in_channels = annotated_graph.x.shape[1]
    num_relations = len(set(annotated_graph.edge_type.tolist()))
    hidden_channels = 256
    out_channels = 256
    print(f"\nGraph properties:")
    print(f"  Input features: {in_channels}")
    print(f"  Number of relations: {num_relations}")
    print(f"  Hidden channels: {hidden_channels}")
    print(f"  Output channels: {out_channels}\n")


    # Define model factory for RGCN
    def create_rgcn_model():
        # return RGCNEncoder(
        #     in_channels=in_channels,
        #     hidden_channels=hidden_channels,
        #     out_channels=out_channels,
        #     num_relations=num_relations,
        #     num_layers=2,
        #     num_bases=100,
        #     dropout=0.5
        # )


        return GCNEncoder(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels)


    # Evaluate on multiple seeds
    results = tdg.evaluate_all(
        kg_name="GT2KG_kg",
        model_factory=create_rgcn_model,
        init_embd="sentence-transformers/all-MiniLM-L6-v2",
        seeds=[42, 123, 456, 789, 1000],  # 5 seeds for robust evaluation
        splits_dir="datasets/split",
        entities_embd_path="KG_GEN_kg_init_embd/EntitiesEmbedding.pickle",
        edges_embd_path="KG_GEN_kg_init_embd/PredicatesEmbedding.pickle",
        epochs=100,
        patience=20,  # Early stopping patience
        lr=0.01,
        weight_decay=5e-4,
        verbose=True,
        save_results=True,
        results_dir="results/GT2KG_kg/GCN"
    )

    # Print final aggregated results
    print("\n" + "=" * 70)
    print("FINAL RESULTS - RGCN on GT2KG_kg")
    print("=" * 70)
    print(
        f"Test Accuracy: {results['aggregated']['test_acc']['mean']:.4f} ± {results['aggregated']['test_acc']['std']:.4f}")
    print(
        f"Test F1:       {results['aggregated']['test_f1']['mean']:.4f} ± {results['aggregated']['test_f1']['std']:.4f}")
    print(
        f"Val Accuracy:  {results['aggregated']['val_acc']['mean']:.4f} ± {results['aggregated']['val_acc']['std']:.4f}")
    print(
        f"Val F1:        {results['aggregated']['val_f1']['mean']:.4f} ± {results['aggregated']['val_f1']['std']:.4f}")
    print(
        f"Best Epoch:    {results['aggregated']['best_epoch']['mean']:.1f} ± {results['aggregated']['best_epoch']['std']:.1f}")
    print("=" * 70)

