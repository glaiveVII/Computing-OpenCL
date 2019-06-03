/* TP Image Open CL Julien PELEGRI et Corentin Giroud */


// PREMIER PROGRAMME ###################################################################
// PREMIER FILTRE MOYENNEUR ON MOYENNE SUR LA VALEUR DE L'ENTIER filter_size
// kernel for mean filter
kernel void mean_filter(__global const unsigned char *imageInput,
           __global       unsigned char *imageOutput,
           int width, int height, int filter_size)
{
  // Get the index of the current element to be processed
  int2 coord = (int2)(get_global_id(0), get_global_id(1));

  int x, y;
  int rowOffset = coord.y * width * 4;
  int index = rowOffset + coord.x * 4;

  // on initialize les coefficient de notre image finale
  float p0 = 0.0;
  float p1 = 0.0;
  float p2 = 0.0;
  float p3 = 0.0;

  // input à modifier ici ///////////////////
  //int filter_size = 10;

  if (coord.x < width && coord.y < height)
  {
    for (x=coord.x-filter_size; x<coord.x+1+filter_size; x++)
    {
      for (y=coord.y-filter_size; y<coord.y+filter_size+1; y++)
      {
        p0 += imageInput[y*width*4+ x*4];
        p1 += imageInput[y*width*4+ x*4+1];
        p2 += imageInput[y*width*4+ x*4+2];
        p3 += imageInput[y*width*4+ x*4+3];
      }
    }
      imageOutput[index]     = p0/((2*filter_size*1)*(2*filter_size*1));
      imageOutput[index + 1] = p1/((2*filter_size*1)*(2*filter_size*1));
      imageOutput[index + 2] = p2/((2*filter_size*1)*(2*filter_size*1));
      imageOutput[index + 3] = p3/((2*filter_size*1)*(2*filter_size*1));

  }

}


// DEUXIEME PROGRAMME ###############################################################################
// VERSION NON OPTIMISÉE DU FILTRE GAUSSIEN
// kernel for gauss filter
kernel void gauss_filter(__global const unsigned char *imageInput,
           __global       unsigned char *imageOutput,
           int width, int height,
            int filter_size, float sigma)
{
  // Get the index of the current element to be processed
  int2 coord = (int2)(get_global_id(0), get_global_id(1));

  int x, y;



  int rowOffset = coord.y * width * 4;
  int index = rowOffset + coord.x * 4;

  // On fait 0.0f pour forcer le typage à un float et eviter les problemes
  float a = 0.0f;
  float e = 0.0f;
  float norm = 0.0f;
  a = 1/sqrt(2.0f*3.14f*100.0f*100.0f);
  float p0 = 0.0;
  float p1 = 0.0;
  float p2 = 0.0;
  float p3 = 0.0;



  if (coord.x < width && coord.y < height)
  {
        imageOutput[index  ] = 0;
        imageOutput[index+1] = 0;
        imageOutput[index+2] = 0;
        imageOutput[index+3] = 0;
    for (x=coord.x-filter_size; x<coord.x+1+filter_size; x++)
    {
      for (y=coord.y-filter_size; y<coord.y+filter_size+1; y++)
      {
        e = exp(-((coord.x-x)*(coord.x-x)+(coord.y-y)*(coord.y-y))/(2.0f*sigma*sigma));
        p0 += a*e*imageInput[y*width*4+ x*4  ];
        p1 += a*e*imageInput[y*width*4+ x*4+1];
        p2 += a*e*imageInput[y*width*4+ x*4+2];
        p3 += a*e*imageInput[y*width*4+ x*4+3];
        norm = norm+a*e;

      }
    }
    imageOutput[index    ] = p0/norm;
    imageOutput[index + 1] = p1/norm;
    imageOutput[index + 2] = p2/norm;
    imageOutput[index + 3] = p3/norm;
  }
}



// TROISIEME PROGRAMME ##################################################################################################
// PREMIERE OPTIMISATION DU GAUSSIEN
// ICI on fait le calcul de a = 1/sqrt(2.0f*3.14*sigma*sigma) dans le programme principal (le .ccp dans etape 8)
// kernel for gauss filter1
kernel void gauss_filter1(__global const unsigned char *imageInput,
           __global       unsigned char *imageOutput,
           int width, int height,
            int filter_size, float sigma, float a)
{
  // Get the index of the current element to be processed
  int2 coord = (int2)(get_global_id(0), get_global_id(1));

  int x ,y;
  int rowOffset = coord.y * width * 4;
  int index = rowOffset + coord.x * 4;

  float e = 0.0f;
  float norm = 0.0f;
  float p0 = 0.0;
  float p1 = 0.0;
  float p2 = 0.0;
  float p3 = 0.0;

  if (coord.x < width && coord.y < height)
  {
    for (x=coord.x-filter_size; x<coord.x+1+filter_size; x++)
    {
      for (y=coord.y-filter_size; y<coord.y+filter_size+1; y++)
      {
        e = exp(-((coord.x-x)*(coord.x-x)+(coord.y-y)*(coord.y-y))/(2.0f*sigma*sigma));
        p0  += a*e*imageInput[y*width*4+ x*4  ];
        p1  += a*e*imageInput[y*width*4+ x*4+1];
        p2  += a*e*imageInput[y*width*4+ x*4+2];
        p3  += a*e*imageInput[y*width*4+ x*4+3];
        norm+= a*e;
      }
    }

    imageOutput[index    ] = p0/norm;
    imageOutput[index + 1] = p1/norm;
    imageOutput[index + 2] = p2/norm;
    imageOutput[index + 3] = p3/norm;
  }

}


// QUATRIEME PROGRAMME #########################################################################################
// DEUXIEME OPTIMISATION DU GAUSSIEN
// On calcul les exponentielles directement dans le programme principal
// kernel for gauss filter2
kernel void gauss_filter2(__global const unsigned char *imageInput,
           __global       unsigned char *imageOutput,
           int width, int height,
            int filter_size, float sigma, float a , __global float *e)
{
  // Get the index of the current element to be processed
  int2 coord = (int2)(get_global_id(0), get_global_id(1));

  int x ,y;
  int rowOffset = coord.y * width * 4;
  int index = rowOffset + coord.x * 4;

  float norm = 0.0f;
  float p0 = 0.0;
  float p1 = 0.0;
  float p2 = 0.0;
  float p3 = 0.0;

  if (coord.x < width && coord.y < height)
  {

    for (x=coord.x-filter_size; x<coord.x+1+filter_size; x++)
    {
     for (y=coord.y-filter_size; y<coord.y+filter_size+1; y++)
      {
        p0  += a*e[x-coord.x+filter_size]*e[y-coord.y+filter_size]*imageInput[y*width*4+ x*4  ];
        p1  += a*e[x-coord.x+filter_size]*e[y-coord.y+filter_size]*imageInput[y*width*4+ x*4+1];
        p2  += a*e[x-coord.x+filter_size]*e[y-coord.y+filter_size]*imageInput[y*width*4+ x*4+2];
        p3  += a*e[x-coord.x+filter_size]*e[y-coord.y+filter_size]*imageInput[y*width*4+ x*4+3];
        norm+= a*e[x-coord.x+filter_size]*e[y-coord.y+filter_size];

      }
    }

    imageOutput[index    ] = p0/norm;
    imageOutput[index + 1] = p1/norm;
    imageOutput[index + 2] = p2/norm;
    imageOutput[index + 3] = p3/norm;
  }

}



// CINQUIEME PROGRAMME ###########################################################################################
// TROISIEME OPTIMISATION DU GAUSSIEN
// On charge directement les valeurs de e dans la memoire locale pour avoir un acces plus rapide
// kernel for gauss filter3
kernel void gauss_filter3(__global const unsigned char *imageInput,
           __global       unsigned char *imageOutput,
           int width, int height,
           int filter_size, float sigma, float a , __constant float *e)
{
  // Get the index of the current element to be processed
  int2 coord = (int2)(get_global_id(0), get_global_id(1));

  int x ,y;
  int rowOffset = coord.y * width * 4;
  int index = rowOffset + coord.x * 4;

  float norm = 0.0f;
  float p0 = 0.0;
  float p1 = 0.0;
  float p2 = 0.0;
  float p3 = 0.0;

  if (coord.x < width && coord.y < height)
  {
    for (x=coord.x-filter_size; x<coord.x+1+filter_size; x++)
    {
      for (y=coord.y-filter_size; y<coord.y+filter_size+1; y++)
      {
        p0  += a*e[x-coord.x+filter_size]*e[y-coord.y+filter_size]*imageInput[y*width*4+ x*4  ];
        p1  += a*e[x-coord.x+filter_size]*e[y-coord.y+filter_size]*imageInput[y*width*4+ x*4+1];
        p2  += a*e[x-coord.x+filter_size]*e[y-coord.y+filter_size]*imageInput[y*width*4+ x*4+2];
        p3  += a*e[x-coord.x+filter_size]*e[y-coord.y+filter_size]*imageInput[y*width*4+ x*4+3];
        norm+= a*e[x-coord.x+filter_size]*e[y-coord.y+filter_size];

      }
    }
    imageOutput[index    ] = p0/norm;
    imageOutput[index + 1] = p1/norm;
    imageOutput[index + 2] = p2/norm;
    imageOutput[index + 3] = p3/norm;
  }

}


// Le SIXIEME programme est mis en commentaire car sinon il fait planter la compilation


// SIXIEME PROGRAMME ################################################################################
// QUATRIEME OPTIMISATION DU GAUSSIEN
// VECTORISATION DES CALCULS : PROGRAMME BIEN COMPLIQUÉ À REALISER ET N'AMELIORE PAS LE TEMPS D'EXECUTION



// kernel for gauss filter4
kernel void gauss_filter4(__global const  uchar4 * imageInput,
           __global      uchar4 *imageOutput,
           int width, int height, int filter_size,
            float sigma, float a , __constant float *e)
{
  // Get the index of the current element to be processed
  int2 coord = (int2)(get_global_id(0), get_global_id(1));

  int x ,y;
  int rowOffset = coord.y * width;
  int index = rowOffset + coord.x;
  float4 norm=(float4)0.0;

  // on cree le float4 et on l'initialise à 0
  float4 b = (float4)(0.0,0.0,0.0,0.0);
  //float4 b = (float4) 0.0;
  //float b = (float)0.0;


  if (coord.x < width && coord.y < height)
  {
    for (x=coord.x-filter_size; x<coord.x+1+filter_size; x++)
    {
      for (y=coord.y-filter_size; y<coord.y+filter_size+1; y++)
      {
        b += a*e[x-coord.x+filter_size]*e[y-coord.y+filter_size]*convert_float4(imageInput[y*width+ x]);
        norm+= a*e[x-coord.x+filter_size]*e[y-coord.y+filter_size];
      }
    }

    //imageOutput[index].x = convert_uchar4(b.x/norm.x);
    //imageOutput[index].y = convert_uchar4(b.y/norm.y);
    //imageOutput[index].z = convert_uchar4(b.z/norm.z);
    //imageOutput[index].w = b.w/norm.w;

    // on normalise la grille finale
    b = b/norm;
    // on remet la transparence comme avant
    b.w = imageInput[index].w;
    // on met l'image de sortie au format
    imageOutput[index] = convert_uchar4(b);



    //imageOutput[index].w = 0;



  }

}

















