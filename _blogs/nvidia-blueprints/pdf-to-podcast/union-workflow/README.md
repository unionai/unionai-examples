# Union Workflow - Nvidia PDF to Podcast Blueprint

## Overview

NVIDIA Blueprint Agents are powerful tools for creating and deploying generative AI applications. They consist of microservices and AI agents orchestrated to deliver specific AI workflows. Union makes **productionizing NVIDIA Blueprint workflows** simple, scalable, and efficient. Developers can focus on business logic while Union takes care of compute and infrastructure. You can also host NIMs locally in Union with the [NIM integration](https://www.union.ai/blog-post/union-powers-faster-end-to-end-ai-application-deployment-using-nvidia-nim).

NVIDIA [launchables](https://developer.nvidia.com/blog/one-click-deployments-for-the-best-of-nvidia-ai-with-nvidia-launchables/?nvid=nv-int-tblg-708008) provide a quick way to deploy Blueprints on pre-configured setups, however, scaling to multiple users or achieving enterprise-grade orchestration requires additional effort.

How Union supports operationalizing NVIDIA Blueprints:

- **Customization**: Blueprints are customizable, but understanding the architecture and scattered boilerplate code can be time-intensive.
- **Mapping**: Developers need to spend significant time unraveling how services interact to fully grasp the workflow as there's no single-pane view.
- **Scaling requirements**: Handling multiple requests or deploying at scale demands further infrastructure setup beyond what Blueprints offer out-of-the-box.
- **Error handling**: The error handling can be simplified and made more sophisticated.

## How Union Can Support and Simplify NVIDIA Blueprints Orchestration

We propose leveraging Union platform which integrates seamlessly with the architecture and requirements of NVIDIA Blueprints, simplifying scaling, monitoring, and maintenance.

### Key Benefits of Using Union

1. **Microservices as Tasks**
   Union enables translating blueprint microservices into Flyte tasks. Instead of setting up API services, tasks can run in Kubernetes pods (or shared pods with Union actors), with independent scaling and resource configurations. No APIs—just clean Python functions. Tasks can also be executed independently, just like the microservices they replace.
   For example, in the PDF-to-podcast workflow, the monologue and dialogue are separate workflows that can be run independently. With Union, you can modularize your code while maintaining a single-pane view of the entire workflow, providing both flexibility and clarity.
2. **Built-in Infrastructure**
   With Union, background jobs, queuing, and storage (Redis/Celery/MinIO, etc.) are handled out-of-the-box. It offers multi-tenancy, concurrent workflow support, and automatic data storage in blob storages.
3. **Monitoring and Logging**

   - Automatic workflow versioning.
   - Built-in **data lineage** tracks the flow of data within workflows, forming an organizational graph for observability.
   - Real-time logging with integration options for custom loggers, eliminating the need for custom trackers.
   - Monitoring services like Jaeger are unnecessary, as microservices are translated into Flyte tasks, simplifying the architecture.

4. **Scalability**
   Union is production-grade and intuitive for scaling. It supports parallel workflows and tracks data and triggers, making it an ideal choice for high-throughput workloads like podcast generation. Tasks can be independently scaled, eliminating the need to define microservices that communicate via APIs. This simplifies the architecture while maintaining flexibility and performance.
5. **Parallelism**
   Nested parallel workflows scale up to 100,000 tasks. For example, summarizing PDFs, processing segments, and generating markdown can run simultaneously using map tasks for significant speedups.
6. **Secrets Management**
   Define and manage secrets directly in the SDK without relying on external secrets managers.
7. **Human-in-the-loop**
   Workflows can include human inputs. In the PDF-to-podcast example, users can visualize speaker options in a Deck and select participants without providing IDs.
8. **Error Handling and Retries**
   Union’s built-in retries handle transient errors, such as external service downtimes, automatically (given we raise flyte-specific retry error). This eliminates boilerplate code and simplifies error management.
9. **Simplified Image Management**
   With ImageSpec, define images directly in Python—no need for Dockerfiles. This makes customization quick and straightforward.
10. **Caching**
    Cache outputs of tasks to reuse results for identical inputs, dramatically improving execution speed and efficiency.
11. **Secure**
    Union is SOC II compliant, ensuring robust data protection and compliance with security standards.

Union makes **productionizing NVIDIA Blueprint workflows** simple, scalable, and efficient. Developers can focus on business logic while Union takes care of compute and infrastructure. You can also host NIMs locally in Union with the [NIM integration](https://www.union.ai/blog-post/union-powers-faster-end-to-end-ai-application-deployment-using-nvidia-nim).

A Union workflow might look like a simple Python script, but behind the scenes, it handles all the heavy lifting of infrastructure and compute orchestration.

All the code needed to run the PDF-to-podcast NVIDIA blueprint as a Union workflow is available in this directory. Take a look to see how clean, customizable, and straightforward it is!
And here's NVIDIA's implementation: https://github.com/NVIDIA-AI-Blueprints/pdf-to-podcast.

## Execution

1. [Sign up for Union Serverless](https://signup.union.ai/).
2. Navigate to the UI at [serverless.union.ai](https://serverless.union.ai/).
3. Install Union: `pip install union`
4. Create the necessary secrets:

   - `union create secret nvidia-build-api-key`
   - `union create secret elevenlabs-api-key`

5. Execute the workflow: `union run --copy all --remote workflow.py pdf_to_podcast`
