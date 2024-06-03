// https://k6.io/docs/examples/http-authentication/#aws-signature-v4-authentication-with-the-k6-jslib-aws

import http from 'k6/http';
import { AWSConfig, SignatureV4 } from 'https://jslib.k6.io/aws/0.8.0/aws.js'
import { check } from 'k6';

const ENDPOINT_NAME = __ENV.ENDPOINT_NAME;
const INFERENCE_COMPONENT = __ENV.INFERENCE_COMPONENT || undefined;
const REGION = __ENV.AWS_REGION || 'us-east-1';
const AWS_ACCESS_KEY_ID = __ENV.AWS_ACCESS_KEY_ID;
const AWS_SECRET_ACCESS_KEY = __ENV.AWS_SECRET_ACCESS_KEY;
const AWS_SESSION_TOKEN = __ENV.AWS_SESSION_TOKEN;

const vu = __ENV.VU || 1;

console.log("ENDPOINT_NAME: " + ENDPOINT_NAME)
console.log("REGION: " + REGION)


const awsConfig = new AWSConfig({
  region: REGION,
  accessKeyId: AWS_ACCESS_KEY_ID,
  secretAccessKey: AWS_SECRET_ACCESS_KEY,
  sessionToken: AWS_SESSION_TOKEN,
});

export const options = {
  thresholds: {
    http_req_failed: ['rate<0.1'],
  },
  scenarios: {
    test: {
      executor: 'constant-vus',
      duration: '180s',
      vus: vu,
    },
  },
};


export default function () {
  // Load ShareGPT random example
  const sample = '{"inputs":[{"name":"prompt","shape":[1,1],"datatype":"BYTES","parameters":{"binary_data_size":24}}],"outputs":[{"name":"generated_image","parameters":{"binary_data":true}}]}\x14\x00\x00\x00cute dragon creature';

  /**
   * Create a signer instance with the AWS credentials.
   * The signer will be used to sign the request.
   */
  const signer = new SignatureV4({
    service: 'sagemaker',
    region: awsConfig.region,
    credentials: {
      accessKeyId: awsConfig.accessKeyId,
      secretAccessKey: awsConfig.secretAccessKey,
      sessionToken: awsConfig.sessionToken,
    },
  });

  /**
   * Use the signer to prepare a signed request.
   * The signed request can then be used to send the request to the AWS API.
   * https://k6.io/docs/javascript-api/jslib/aws/signaturev4/
   */
  const headers = {
    'Content-Type': 'application/vnd.sagemaker-triton.binary+json;json-header-size=173',
  };
  if (INFERENCE_COMPONENT) {
    headers['X-Amzn-SageMaker-Inference-Component'] = INFERENCE_COMPONENT
  }

  const signedRequest = signer.sign({
    method: 'POST',
    protocol: 'https',
    hostname: `runtime.sagemaker.${REGION}.amazonaws.com`,
    path: `/endpoints/${ENDPOINT_NAME}/invocations`,
    headers: headers,
    body: sample,
    uriEscapePath: false,
    applyChecksum: false,
  });

  const res = http.post(signedRequest.url, signedRequest.body, { headers: signedRequest.headers });

  check(res, {
    'Post status is 200': (r) => res.status === 200,
  });
}
