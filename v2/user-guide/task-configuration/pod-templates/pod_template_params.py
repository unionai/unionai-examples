# {{docs-fragment pod-template-params}}
pod_template = flyte.PodTemplate(
    primary_container_name: str = "primary",
    pod_spec: Optional[V1PodSpec] = None,
    labels: Optional[Dict[str, str]] = None,
    annotations: Optional[Dict[str, str]] = None
)
# {{/docs-fragment pod-template-params}}